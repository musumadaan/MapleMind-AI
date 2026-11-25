"""Contract test: the real langchain_ollama.ChatOllama client, through the full LangGraph pipeline, against a
protocol-faithful fake Ollama server. Verifies request format, generation options, streaming/usage parsing."""
import fake_ollama
from maplemind.app import MapleMind
from maplemind.config import load_settings


def test_full_pipeline_through_chat_ollama():
    srv, url = fake_ollama.start()
    try:
        cfg = load_settings()
        cfg["llm"].update(provider="ollama", model="qwen2.5:1.5b-instruct", base_url=url)
        mm = MapleMind(cfg, trace=False)
        from langchain_ollama import ChatOllama
        assert isinstance(mm.llm, ChatOllama)
        r = mm.run("How long do people wait for community mental health counselling?")
        chat_calls = [q for q in fake_ollama.REQUESTS if q["path"] == "/api/chat"]
        assert chat_calls, "no request reached the Ollama API"
        body = chat_calls[-1]["body"]
        assert body["model"] == "qwen2.5:1.5b-instruct"
        assert body["options"]["temperature"] == cfg["llm"]["temperature"] and body["options"]["seed"] == cfg["llm"]["seed"]
        assert body["options"]["num_predict"] == cfg["llm"]["max_new_tokens"]
        assert body["messages"][0]["role"] == "system" and "EVIDENCE:" in body["messages"][-1]["content"]
        assert r["status"] == "answered" and r["citations"], r["answer"]
        comp = [s for s in mm.tracer.history[-1].spans if s.name == "composer"][0]
        assert comp.tokens_in > 0 and comp.tokens_out > 0            # usage parsed from Ollama's final chunk
    finally:
        srv.shutdown()


def test_llm_judge_through_chat_ollama():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from judge import judge
    from maplemind.llm.lc import build_chat_model
    srv, url = fake_ollama.start()
    try:
        llm = build_chat_model({"provider": "ollama", "model": "llama3.1:8b", "base_url": url, "temperature": 0.0, "seed": 7})
        out = judge(llm, "q?", "The median wait was 25 days [1].", [{"n": 1, "text": "The median wait was 25 days."}], {"supported_ratio": 1.0}, True)
        assert out["groundedness"] == 5 and out["safety"] == 5
    finally:
        srv.shutdown()


def test_cohen_kappa_and_calibration(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    from judge import calibrate, cohen_kappa
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0])) < 1e-9          # chance-level agreement
    csv = tmp_path / "h.csv"
    csv.write_text("case_id,rater,groundedness\nA,r1,5\nB,r1,2\n")
    rows = [{"case_id": "A", "judge": {"groundedness": 5}}, {"case_id": "B", "judge": {"groundedness": 5}}]
    c = calibrate(rows, csv)
    assert c["n"] == 2 and c["agreement"] == 0.5
