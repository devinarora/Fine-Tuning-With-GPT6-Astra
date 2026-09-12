import pytest

from fr_en_ft.benchmark import plan_steps
from fr_en_ft.config import Config
from fr_en_ft.evaluate import diagnose, paired_bootstrap, summarize


def records(hypotheses):
    references = ["The little cat sleeps on the warm chair.", "Tomorrow we will meet at the railway station."]
    return [{"id": str(i), "corpus": "books" if i == 0 else "opus", "target": ref,
             "hypothesis": hyp, "nll_sum": (i + 1) ** 2 * 2, "target_tokens": i + 1,
             "empty": False, "copied_source": False, "repeated_trigram": False,
             "number_mismatch": False, "output_limit_hit": False, "length_ratio": 1}
            for i, (ref, hyp) in enumerate(zip(references, hypotheses))]


def test_corpus_metrics_and_token_weighted_nll():
    rows = records(["The little cat sleeps on the warm chair.", "Tomorrow we will meet at the railway station."])
    scores, signatures = summarize(rows)
    assert scores["bleu"] == pytest.approx(100)
    assert scores["chrf++"] == pytest.approx(100)
    assert scores["ter"] == 0
    assert scores["nll"] == pytest.approx(10 / 3)
    assert "tok:13a" in signatures["bleu"]
    assert "nw:2" in signatures["chrf++"]


def test_paired_bootstrap_keeps_alignment_and_metric_direction():
    bad = records(["Bad translation here.", "Nothing matches the source."])
    good = records(["The little cat sleeps on the warm chair.", "Tomorrow we will meet at the railway station."])
    result = paired_bootstrap(bad, good, 10, 42)["intervals_95"]
    assert result["bleu"]["low"] > 0
    assert result["ter"]["high"] < 0
    with pytest.raises(ValueError, match="ordered"):
        paired_bootstrap(bad, list(reversed(good)), 10, 42)
    zero = paired_bootstrap(good, good, 10, 42)["intervals_95"]
    assert all(ci == {"low": 0.0, "high": 0.0} for ci in zero.values())


def test_diagnostics_are_heuristics():
    flags = diagnose("Il a 12 chats.", "He has 12 cats.", "Il a 12 chats.")
    assert flags["copied_source"]
    assert not flags["number_mismatch"]
    assert diagnose("x", "one", "")["empty"]


def test_step_budget_includes_evaluation_and_checkpoint_overhead():
    config = Config()
    bench = {"micro_batch_size": 2, "gradient_accumulation_steps": 8,
             "update_seconds_p75": 1, "validation_seconds_estimate": 100,
             "checkpoint_seconds": 5, "gradient_checkpointing": False, "precision": "bf16"}
    plan = plan_steps(config, 20000, bench, remaining=3000, final_reserve=1800)
    assert 0 < plan["max_steps"] < 2500
    assert plan["predicted_end_to_end_remaining_seconds"] <= 3000
    with pytest.raises(RuntimeError, match="No training"):
        plan_steps(config, 20000, bench, remaining=1800, final_reserve=1800)
