from maplemind.skills.screening import Screener, parse_answer


def run(answers, budget=12):
    sc = Screener(["phq9", "gad7"], budget)
    s = sc.new()
    for a in answers:
        s, err = sc.answer(s, a)
        assert err is None
        if s.done:
            break
    return s, sc.summary(s)


def test_parse_answers():
    assert parse_answer("2") == 2 and parse_answer("nearly every day") == 3 and parse_answer("not at all") == 0
    assert parse_answer("banana") is None


def test_negative_brief_screens_stop_early():
    s, summ = run(["0", "1", "0", "1"])
    assert s.done and s.asked == 4 and not summ["partial"]


def test_positive_phq_runs_full_then_budget_stops():
    s, summ = run(["2", "2", "1", "2", "1", "2", "1", "1", "0", "3", "2", "2", "2", "2"])
    assert summ["results"]["phq9"]["total"] == 12 and summ["results"]["phq9"]["band"] == "moderate"
    assert s.asked == 12 and summ["partial"]


def test_item9_triggers_safety():
    s, _ = run(["3", "3", "2", "2", "2", "2", "2", "2", "1"])
    assert s.safety_trigger and s.done
