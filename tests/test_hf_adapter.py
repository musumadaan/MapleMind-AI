"""The fine-tuned LoRA adapter runs inside the full LangGraph app through LangChain (MapleChatModel over HF + PEFT)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE, ADAPTER = ROOT / "finetune/outputs/tiny-qwen2-base", ROOT / "finetune/outputs/tiny-qwen2-lora-balanced"


@pytest.mark.skipif(not ADAPTER.exists(), reason="run finetune/make_tiny_base.py and train_lora.py first")
def test_adapter_inside_graph():
    pytest.importorskip("peft")
    from maplemind.app import MapleMind
    from maplemind.config import load_settings
    cfg = load_settings()
    cfg["llm"].update(provider="hf", hf_model_id=str(BASE), adapter_path=str(ADAPTER), max_new_tokens=40, temperature=0.0)
    mm = MapleMind(cfg, use_adapter=True, trace=False)
    assert mm.llm.provider.label.endswith("+lora")
    r = mm.run("How much exercise should adults get each week?")
    comp = [s for s in mm.tracer.history[-1].spans if s.name == "composer"][0]
    assert comp.tokens_out > 0 and r["status"] in ("answered", "abstained")   # verifier gates whatever the model says
