import time
from types import SimpleNamespace

from transformers import TrainerControl

from fr_en_ft.config import Config
from fr_en_ft.train import Progress


def progress(remaining=1000, saved=None):
    run = SimpleNamespace(config=Config(), remaining=remaining)
    plan = {"final_reserve_seconds": 100}
    benchmark = {"validation_seconds_estimate": 20, "checkpoint_seconds": 3}
    return Progress(run, plan, benchmark, saved)


def test_periodic_save_budget_stop_and_interrupt_boundaries():
    callback = progress()
    callback.last_save = time.monotonic() - 300
    state = SimpleNamespace(global_step=10, max_steps=100)
    control = callback.on_step_end(None, state, TrainerControl())
    assert control.should_save
    assert not control.should_training_stop
    callback.run.remaining = 150
    control = callback.on_step_end(None, state, TrainerControl())
    assert control.should_training_stop and control.should_save and control.should_evaluate
    assert callback.stop_reason == "time_budget"
    callback.interrupt = True
    control = callback.on_step_end(None, state, TrainerControl())
    assert control.should_training_stop and control.should_save and not control.should_evaluate
    assert callback.stop_reason == "interrupted"


def test_early_stopping_counter_survives_resume():
    callback = progress(saved={"best": 50, "bad_evaluations": 2})
    control = callback.on_evaluate(None, None, TrainerControl(), {"eval_chrf": 49, "eval_loss": 1.5})
    assert control.should_training_stop
    assert callback.stop_reason == "early_stopping"
    restored = progress(saved=callback.snapshot())
    restored.on_evaluate(None, None, TrainerControl(), {"eval_chrf": 51, "eval_loss": 1.4})
    assert restored.bad_evaluations == 0
    assert restored.best == 51
