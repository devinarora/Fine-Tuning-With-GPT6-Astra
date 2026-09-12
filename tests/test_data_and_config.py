import pytest

from fr_en_ft.config import from_dict
from fr_en_ft.data import assert_disjoint, books_split, clean_pair, normalize
from fr_en_ft.state import Run, read_json


def test_french_is_input_and_accents_are_preserved():
    row, reason = clean_pair({"translation": {"fr": "  Cafe\u0301\n ici. ", "en": " Cafe here. "}}, "books", "train", 0)
    assert reason is None
    assert row["source"] == "Café ici."
    assert row["target"] == "Cafe here."
    assert normalize("  ÉTÉ \t français ") == "ÉTÉ français"


def test_bad_pairs_are_countable_exclusions():
    assert clean_pair({"translation": {"fr": "x", "en": ""}}, "books", "train", 0)[1] == "empty"
    assert clean_pair({"translation": {"fr": "x", "en": "y" * 100}}, "books", "train", 0)[1] == "character_ratio"
    assert clean_pair({}, "books", "train", 0)[1] == "invalid_schema"


def test_source_grouping_prevents_alternate_translation_leakage():
    a, _ = clean_pair({"translation": {"fr": "Bonjour.", "en": "Hello."}}, "books", "train", 1)
    b, _ = clean_pair({"translation": {"fr": "Bonjour.", "en": "Good morning."}}, "opus", "test", 2)
    assert a["pair_hash"] != b["pair_hash"]
    assert books_split(a["source_hash"], 42) == books_split(b["source_hash"], 42)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint({"train": [a], "test": [b]})


def test_config_rejects_misspellings_mutable_revisions_and_invalid_batches():
    with pytest.raises(ValueError, match="Unknown"):
        from_dict({"training": {"learnng_rate": 1e-5}})
    with pytest.raises(ValueError, match="commit"):
        from_dict({"assets": {"model_revision": "main"}})
    with pytest.raises(ValueError, match="divisible"):
        from_dict({"training": {"micro_batch_size": 3}})


def test_run_identity_and_explicit_budget_extension(tmp_path):
    config = from_dict({"runtime": {"run_root": str(tmp_path)}})
    run = Run.create(config, "example")
    with pytest.raises(FileExistsError):
        Run.create(config, "example")
    run.state["spent_seconds"] = 7000
    run.state["extra_seconds"] = 300
    run.heartbeat()
    assert Run(run.path).remaining == 500
    assert read_json(run.path / "config.json") == config.to_dict()
    with pytest.raises(ValueError):
        Run.create(config, "../escape")
