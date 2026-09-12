# Local French-to-English fine-tuning

This project fine-tunes a pretrained French-to-English translation model and compares it with the original model. The workflow records how data was selected, how training progressed, and whether translation quality changed.

## Run the workflow

Open a WSL terminal in the repository root. Complete the [environment setup](environments.md) first. The examples use `fr-en-ft` as a generic environment name.

```bash
conda activate fr-en-ft
python -m fr_en_ft doctor
python -m pytest -q
python -m fr_en_ft run --config configs/smoke.yaml --run-id smoke-001
python -m fr_en_ft run --config configs/local.yaml --run-id local-001
```

Tests run exclusively in WSL. The test configuration rejects other operating systems. The run IDs in these examples are placeholders; choose a new ID for each experiment.

| Profile | Training pairs | Validation pairs | Test pairs | Purpose |
|---|---:|---:|---:|---|
| `smoke.yaml` | 64 | 8 | 8 | Exercise every stage with four training updates |
| `local.yaml` | 20,000 | 500 | 1,000 | Run the normal experiment |

Each split contains equal numbers of Books and OPUS-100 examples. The normal profile limits inputs and references to 256 Marian tokens. Generation allows up to 256 tokens after the decoder's start token. Smoke-run scores are only a functional check.

By default, run artifacts go under `~/fr-en-ft/runs` and downloads under `~/fr-en-ft/cache`. Both paths are configurable. The `~` refers to the current user's home directory; no account-specific path is needed in the configuration.

## Stages and artifacts

| Stage | Work | Main artifacts |
|---|---|---|
| `download` | Fetch pinned model, scorer, and only French-English dataset files | `assets.json` and shared cache |
| `prepare` | Normalize, filter, split, deduplicate, tokenize, and materialize samples | `prepared/`, `data_manifest.json` |
| `benchmark` | Probe full-length memory, time updates and generation, measure checkpoint I/O | `benchmark.json` |
| `baseline` | Generate translations and evaluate original weights | `baseline_metrics.json`, `baseline_predictions.json` |
| `train` | Freeze a time-aware step budget, train, validate, and checkpoint | `training_plan.json`, `training_result.json`, `checkpoints/` |
| `final` | Evaluate the validation-best fine-tuned checkpoint | `final_metrics.json`, `final_predictions.json` |
| `report` | Compare scores, bootstrap differences, and display two fixed examples | `report.md`, `comparison.csv`, `confidence_intervals.json`, `examples.json`, `learning-curves.png` |

Each stage runs in a separate process to release model memory between stages. A filesystem lock prevents concurrent writes to the same run. JSON artifacts and stage-completion records use atomic writes.

To work through the stages individually:

```bash
python -m fr_en_ft init --config configs/local.yaml --run-id inspect-001
python -m fr_en_ft download --run-dir ~/fr-en-ft/runs/inspect-001
python -m fr_en_ft prepare --run-dir ~/fr-en-ft/runs/inspect-001
python -m fr_en_ft benchmark --run-dir ~/fr-en-ft/runs/inspect-001
python -m fr_en_ft run --resume ~/fr-en-ft/runs/inspect-001
```

Stages enforce prerequisites. Completed stages are skipped. To change a configuration, start a new run rather than editing a saved run.

## Data decisions

The loader reads revision-pinned Parquet files. French input comes from `translation.fr`, and English labels come from `translation.en`. The dataset configuration name `en-fr` does not set the translation direction.

The cleaner applies Unicode NFC and collapses whitespace while preserving case, accents, and punctuation. It removes malformed records, empty translations, character-length ratios above 8, and duplicate pairs. Identical normalized French cannot cross split boundaries, including across datasets.

OPUS-100 uses its official splits. Books assigns a hash of each normalized French source to a stable split: 5% test, 5% validation, and 90% training. Test membership takes precedence over validation, then training. This rule covers all candidate holdouts, including rows outside the final sample. Alternate translations of the same French source stay in one split.

Seeded hashes order candidates across each source and split. The tokenizer processes candidates until it finds enough eligible examples. Overlong inputs and references are excluded rather than truncated. The manifest reports full-corpus cleaning and candidate token-length filtering separately.

The manifest records row identities, content hashes, split order, and a data fingerprint. One test example per source is selected before training for the two-example display.

Books does not expose reliable book identifiers. Its source-grouped holdout does not guarantee unseen-book evaluation. OPUS-100 has no reliable dialogue labels, so its scores are labeled mixed-domain rather than spoken-language scores.

## Training and budget

The default settings use AdamW, a learning rate of `2e-5`, weight decay of `0.01`, gradient clipping at `1.0`, and 5% warmup. The effective batch size is 16. Calibration starts with a microbatch of two and eight accumulation steps, then tries smaller microbatches and gradient checkpointing if needed.

Training loss is normalized over all non-padding target tokens in an accumulated batch. The Trainer subclass applies this denominator explicitly because Marian's forward method does not use it. A regression test compares accumulated gradients with one equivalent batch, including unequal target lengths and ignored padding.

Calibration checks a synthetic batch at the configured token limits, then measures training, generation, and checkpoint I/O. These measurements set the local step budget. Calibration updates are discarded; baseline evaluation and fine-tuning each reload the pinned original weights.

The budget accounts for training updates, validation, checkpoint writes, and final evaluation. Training stops at the smaller of two epochs and the calculated step budget. The fixed test set does not select hyperparameters or checkpoints. Its evaluation duration helps reserve time for final scoring.

The normal profile targets two hours of active execution. Setup, initial downloads, idle time between commands, and interpreter startup are excluded. Finishing a batch, checkpoint, scoring call, or report can cause a small overrun. Other applications can also affect throughput.

Validation runs at a calibrated interval and at the final update. It reports generated BLEU, chrF++, TER, and validation loss. The highest pooled validation chrF++ selects the final checkpoint. Three non-improving validation rounds trigger early stopping.

`events.jsonl` records training progress and local timing diagnostics. The training ETA covers remaining updates; the saved training plan accounts for final evaluation and checkpoint costs separately. Use the [sharing guidance](#sharing-results) when preparing public results.

## Recovery

```bash
python -m fr_en_ft run --resume ~/fr-en-ft/runs/local-001
```

Ctrl+C requests a stop after a complete optimizer update and saves a checkpoint before exiting. A hard termination recovers from the most recent complete checkpoint. Save requests follow the configured interval; completing an update, validation pass, or disk write can delay the save.

Checkpoints contain model weights, tokenizer files, optimizer, scheduler, random states, training control, and the mixed-precision scaler when applicable. Each file has a checksum. A checkpoint becomes available only after all files have been written and synchronized. Retention keeps the latest two complete checkpoints and the validation-best checkpoint.

Resume checks configuration, data, training plan, source-code hashes, and key package versions. It falls back when the newest checkpoint is incomplete or corrupt. Recovery after the final checkpoint does not add a training update. Prediction stages also save completed batches for resumption.

To extend an exhausted budget explicitly:

```bash
python -m fr_en_ft run --resume ~/fr-en-ft/runs/local-001 --extra-seconds 600
```

The extension is logged. It adds execution time without changing the original training schedule or restarting completed stages.

## Translate a short paragraph

```bash
python -m fr_en_ft translate --run-dir ~/fr-en-ft/runs/local-001 --text "Bonjour, le train arrive dans dix minutes."
```

The command rejects oversized input with a clear error. Document segmentation and cross-paragraph context are outside this version.

## Sharing results

Prepare a publication copy from aggregate quality scores in `comparison.csv` and confidence intervals in `confidence_intervals.json`. Include model and dataset revisions, metric settings, and source attribution so readers can interpret the results.

Local artifacts can contain account paths, environment names, run IDs, timestamps, hardware details, and benchmark timings. Exclude those fields from publication copies. Generated `report.md` files also contain timing information and a run ID, so review them before sharing. There is no automatic sanitized-export command.

Keep recovery files intact. Their paths, checksums, and state are needed to resume training. Publish a separate summary rather than editing active checkpoints or copying an entire run directory.

Clear notebook outputs before saving a public copy. Environment inspection, progress logs, and embedded reports can expose local details. The repository's ignore rules cover common generated artifacts, but manually copied files still need review.
