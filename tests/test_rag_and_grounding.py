from maplemind.config import load_settings
from maplemind.rag.ingest import build_index
from maplemind.skills.grounding import check

CFG = load_settings()
R = build_index(CFG)


def test_retrieves_right_doc():
    hits = R.search("How long is the wait for community mental health counselling?")
    assert hits[0].chunk.doc_id == "cihi_mental_health_wait_times"


def test_hybrid_handles_acronyms():
    hits = R.search("PHQ-9 severity bands")
    assert any(h.chunk.doc_id == "screening_instruments_evidence" for h in hits)


def test_numeric_hallucination_caught():
    p = [{"n": 1, "text": "The median wait was 25 days."}]
    assert check("The median wait for counselling was 25 days [1].", p).ok
    assert not check("The median wait for counselling was 40 days [1].", p).ok


def test_uncited_claim_caught():
    p = [{"n": 1, "text": "Obesity rose to 33% among adults."}]
    assert not check("Obesity rose to 33% among Canadian adults recently.", p).ok


def test_invalid_citation_caught():
    p = [{"n": 1, "text": "Obesity rose to 33% among adults."}]
    assert check("Obesity rose to 33% among adults [3].", p).invalid_citations == [3]


def test_vector_db_is_chroma_and_scores_are_similarities():
    import numpy as np
    assert R.store.backend == "chroma" and R.store.info()["count"] == len(R.store.chunks)
    qv = R.embedder.embed(["How long is the wait for counselling?"])[0]
    got = R.store.search(qv, 5)
    exact = sorted(((i, float(R.store.vecs[i] @ qv)) for i in range(len(R.store.chunks))), key=lambda x: -x[1])[:5]
    assert [i for i, _ in got] == [i for i, _ in exact]
    assert np.allclose([s for _, s in got], [s for _, s in exact], atol=1e-4)


def test_metadata_filter():
    hits = R.search("healthy eating plate", where={"topic": "nutrition"})
    assert hits and {h.chunk.doc_id for h in hits} == {"canada_food_guide"}
