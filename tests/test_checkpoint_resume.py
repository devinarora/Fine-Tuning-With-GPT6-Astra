"""Compare actual Trainer state after uninterrupted and interrupted training."""

from datasets import Dataset
import pytest
import torch
from transformers import MarianConfig, MarianMTModel, Seq2SeqTrainingArguments, TrainerCallback, set_seed

from fr_en_ft.checkpoints import AtomicTrainer, inspect_checkpoint, latest_checkpoint


class StopAfterTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
            control.should_save = True
        return control


def make_trainer(path, stop=False, dropout=0.1, use_cuda=False):
    set_seed(123)
    model = MarianMTModel(MarianConfig(vocab_size=32, decoder_vocab_size=32, d_model=16,
                          encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
                          encoder_ffn_dim=32, decoder_ffn_dim=32, max_position_embeddings=32,
                          pad_token_id=0, decoder_start_token_id=0, eos_token_id=2, bos_token_id=1,
                          dropout=dropout, attention_dropout=dropout))
    dataset = Dataset.from_list([{"input_ids": [3 + i % 4, 5, 2], "attention_mask": [1, 1, 1],
                                  "labels": [6 + i % 4, 7, 2]} for i in range(12)])
    args = Seq2SeqTrainingArguments(output_dir=str(path), use_cpu=not use_cuda, bf16=use_cuda, max_steps=4,
                                    per_device_train_batch_size=2, gradient_accumulation_steps=2,
                                    learning_rate=0.001, save_steps=2, seed=123, data_seed=123,
                                    report_to="none", logging_strategy="no", disable_tqdm=True,
                                    optim="adamw_torch", dataloader_num_workers=0)
    return AtomicTrainer(model=model, args=args, train_dataset=dataset, checkpoint_contract="test-contract",
                         callbacks=[StopAfterTwo()] if stop else [])


@pytest.mark.integration
def test_accumulated_gradients_match_one_token_weighted_batch(tmp_path):
    import copy
    trainer = make_trainer(tmp_path / "gradients", dropout=0)
    reference = copy.deepcopy(trainer.model)
    batch = {"input_ids": torch.tensor([[4, 5, 2], [6, 2, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
             "labels": torch.tensor([[7, 8, 2], [9, 2, -100]])}
    expected_loss = reference(**batch).loss
    expected_loss.backward()
    microbatches = [{key: value[i:i + 1] for key, value in batch.items()} for i in range(2)]
    denominator = trainer._get_num_items_in_batch(microbatches, torch.device("cpu"))
    assert denominator.item() == 5
    trainer.current_gradient_accumulation_steps = 2
    observed_loss = sum(trainer.training_step(trainer.model, microbatch, denominator) for microbatch in microbatches)
    torch.testing.assert_close(observed_loss, expected_loss.detach())
    for actual, expected in zip(trainer.model.parameters(), reference.parameters()):
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=1e-7)


@pytest.mark.integration
@pytest.mark.parametrize("use_cuda", [False, True])
def test_resume_matches_uninterrupted_training_and_corruption_falls_back(tmp_path, use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.set_num_threads(2)
    uninterrupted = make_trainer(tmp_path / "continuous", use_cuda=use_cuda)
    uninterrupted.train()
    expected = {k: value.clone() for k, value in uninterrupted.model.state_dict().items()}
    interrupted = make_trainer(tmp_path / "resumed", stop=True, use_cuda=use_cuda)
    interrupted.train()
    checkpoint = latest_checkpoint(tmp_path / "resumed", "test-contract")
    assert inspect_checkpoint(checkpoint)["global_step"] == 2
    resumed = make_trainer(tmp_path / "resumed", use_cuda=use_cuda)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == uninterrupted.state.global_step == 4
    for key, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=1e-6, atol=1e-7, msg=key)
    assert resumed.lr_scheduler.state_dict() == uninterrupted.lr_scheduler.state_dict()
    newest = latest_checkpoint(tmp_path / "resumed", "test-contract")
    file = newest / "scheduler.pt"
    original = file.read_bytes()
    file.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="checksum"):
        inspect_checkpoint(newest, "test-contract")
    assert latest_checkpoint(tmp_path / "resumed", "test-contract") == checkpoint
    with pytest.raises(ValueError, match="mismatch"):
        inspect_checkpoint(checkpoint, "another-configuration")


@pytest.mark.integration
def test_pipeline_resumes_a_finished_checkpoint_without_an_extra_update(tmp_path, monkeypatch):
    from datasets import DatasetDict
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from fr_en_ft.config import from_dict
    from fr_en_ft.state import Run, atomic_json, read_json
    from fr_en_ft.train import train

    tiny = make_trainer(tmp_path / "unused").model
    base = tmp_path / "base-model"
    tiny.save_pretrained(base)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f"t{i}": i for i in range(32)}, unk_token="t3")), pad_token="t0", bos_token="t1", eos_token="t2", unk_token="t3")
    tokenizer.save_pretrained(base)
    config = from_dict({"runtime": {"run_root": str(tmp_path / "runs"), "require_cuda": False, "cpu_threads": 2},
                        "data": {"max_source_tokens": 8, "max_target_tokens": 8},
                        "training": {"micro_batch_size": 2, "effective_batch_size": 4, "max_steps": 2},
                        "evaluation": {"bertscore": False, "batch_size": 2, "beams": 1}})
    run = Run.create(config, "finished")
    run.start_clock()
    atomic_json(run.path / "assets.json", {"paths": {"model": str(base)}})
    ds = Dataset.from_list([{"input_ids": [4, 5, 2], "attention_mask": [1, 1, 1], "labels": [6, 7, 2]} for _ in range(8)])
    DatasetDict(train=ds, validation=ds.select(range(2)), test=ds.select(range(2))).save_to_disk(str(run.path / "prepared"))
    atomic_json(run.path / "data_manifest.json", {"config_fingerprint": config.fingerprint, "data_fingerprint": "tiny-fixture"})
    atomic_json(run.path / "benchmark.json", {"micro_batch_size": 2, "gradient_accumulation_steps": 2,
                "update_seconds_p75": 0.1, "validation_seconds_estimate": 1, "checkpoint_seconds": 1,
                "gradient_checkpointing": False, "precision": "fp32"})
    atomic_json(run.path / "baseline_metrics.json", {"performance": {"generation_and_nll_seconds": 1,
                "semantic_seconds": 0, "model_load_seconds": 0.1}})
    train(run)
    original = read_json(run.path / "training_result.json")
    run.state["completed"].remove("train")
    run.heartbeat()
    (run.path / "training_result.json").unlink()

    def unexpected_update(*args, **kwargs):
        raise AssertionError("An already finished checkpoint must not perform another training update")

    monkeypatch.setattr(AtomicTrainer, "training_step", unexpected_update)
    resumed_run = Run(run.path)
    resumed_run.start_clock()
    train(resumed_run, resume=True)
    restored = read_json(run.path / "training_result.json")
    assert restored["global_step"] == original["global_step"] == 2
    assert restored["best_checkpoint"] == original["best_checkpoint"]
