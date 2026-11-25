"""Build a tiny, randomly initialised Qwen2-architecture model + ChatML tokenizer, locally (no downloads).

Purpose: verify the SFT pipeline end-to-end (chat template -> assistant-only masking -> LoRA on q/k/v/o/gate/up/down ->
Trainer with early stopping -> adapter save -> adapter load for inference -> model matrix) in environments where
Hugging Face is unreachable or there is no GPU. Its outputs are NOT a quality result.

  python finetune/make_tiny_base.py      ->  finetune/outputs/tiny-qwen2-base
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "finetune/outputs/tiny-qwen2-base"
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")   # same format Qwen2.5 uses


def main():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    import torch
    torch.manual_seed(7)
    texts = []
    for split in ("train", "val", "test"):
        for line in open(ROOT / f"finetune/data/{split}.jsonl"):
            texts += [m["content"] for m in json.loads(line)["messages"]]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=3000, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                     bos_token=None, unk_token=None)
    hf_tok.chat_template = CHATML
    cfg = Qwen2Config(vocab_size=len(hf_tok), hidden_size=128, intermediate_size=344, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=True,
                      eos_token_id=hf_tok.convert_tokens_to_ids("<|im_end|>"), pad_token_id=hf_tok.pad_token_id)
    model = Qwen2ForCausalLM(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    hf_tok.save_pretrained(OUT)
    n = sum(p.numel() for p in model.parameters())
    print(f"saved tiny Qwen2 ({n/1e6:.2f}M params, vocab {len(hf_tok)}) to {OUT}")


if __name__ == "__main__":
    main()
