"""QLoRA supervised fine-tuning for MapleMind (assistant-only loss).

  pip install -r finetune/requirements-train.txt
  python finetune/build_sft_dataset.py --mentalchat 400
  python finetune/train_lora.py --config finetune/configs/qwen25_1p5b_qlora.yaml

Implementation notes
- Uses the model's own chat template (identical formatting at train and inference time).
- Labels for system+user tokens are -100: loss is computed only on the assistant answer (citations, abstentions,
  boundaries), which is exactly the behaviour we want to change. Evidence text is never "memorised" as a target.
- Plain HF Trainer + PEFT (no TRL version coupling). Early stopping on validation loss; best adapter is kept.
- Logs train/val loss, peak GPU memory and wall-clock time to <output_dir>/train_metrics.json for the report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune/configs/qwen25_1p5b_qlora.yaml")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: e.g. 5")
    ap.add_argument("--base-model", help="override base_model (HF id or local path)")
    ap.add_argument("--output-dir", help="override output_dir")
    ap.add_argument("--epochs", type=float, help="override train.epochs")
    ap.add_argument("--lr", type=float, help="override train.learning_rate")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.base_model:
        cfg["base_model"] = a.base_model
    if a.output_dir:
        cfg["output_dir"] = a.output_dir
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    if a.lr:
        cfg["train"]["learning_rate"] = a.lr

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    t = cfg["train"]
    set_seed(t["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok.pad_token = tok.pad_token or tok.eos_token
    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
                             bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
                             bnb_4bit_compute_dtype=getattr(torch, q["compute_dtype"])) if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=bnb,
                                                 dtype=getattr(torch, q["compute_dtype"]) if torch.cuda.is_available() else torch.float32)
    if bnb:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=t["gradient_checkpointing"])
    l = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
                                             target_modules=l["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def ids_of(x):   # transformers 5 returns a BatchEncoding from apply_chat_template(tokenize=True)
        return list(x["input_ids"]) if hasattr(x, "keys") else list(x)

    def encode(ex):
        msgs = ex["messages"]
        prompt_ids = ids_of(tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True))
        full_ids = ids_of(tok.apply_chat_template(msgs, tokenize=True))
        assert full_ids[:len(prompt_ids)] == prompt_ids, "chat template is not prefix-stable; masking would be wrong"
        full_ids = full_ids[: t["max_seq_len"]]
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels[: len(full_ids)]}

    d = cfg["data_dir"]
    ds = load_dataset("json", data_files={"train": f"{d}/train.jsonl", "validation": f"{d}/val.jsonl"})
    ds = ds.map(encode, remove_columns=ds["train"].column_names)
    n_sup = sum(sum(1 for x in r["labels"] if x != -100) for r in ds["train"])
    n_all = sum(len(r["labels"]) for r in ds["train"])
    print(f"train examples {len(ds['train'])}, tokens {n_all}, supervised (assistant) tokens {n_sup} ({n_sup / n_all:.1%})")
    pre_eval = None
    import math
    steps_per_epoch = math.ceil(len(ds["train"]) / (t["per_device_batch_size"] * t["gradient_accumulation"]))
    total_steps = a.max_steps if a.max_steps > 0 else math.ceil(steps_per_epoch * t["epochs"])
    warmup_steps = int(t["warmup_ratio"] * total_steps)          # transformers 5 removed `warmup_ratio`
    args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=t["epochs"], max_steps=a.max_steps,
        per_device_train_batch_size=t["per_device_batch_size"], per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"], warmup_steps=warmup_steps, weight_decay=t["weight_decay"],
        logging_steps=5, eval_strategy="steps", eval_steps=t["eval_every_steps"], save_strategy="steps",
        save_steps=t["eval_every_steps"], save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=torch.cuda.is_available() and q["compute_dtype"] == "bfloat16",
        fp16=torch.cuda.is_available() and q["compute_dtype"] == "float16",
        gradient_checkpointing=t["gradient_checkpointing"], report_to="none", seed=t["seed"])
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"])])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pre_eval = trainer.evaluate()["eval_loss"]
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    metrics = {"wall_clock_min": round(wall / 60, 2), "log_history": trainer.state.log_history,
               "best_eval_loss": trainer.state.best_metric, "eval_loss_before_training": pre_eval,
               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "total_params": sum(p.numel() for p in model.parameters()), "base_model": cfg["base_model"],
               "supervised_token_share": round(n_sup / n_all, 4), "steps": trainer.state.global_step,
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output_dir"]) / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
