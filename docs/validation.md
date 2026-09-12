# Recorded evaluation

This page summarizes a previously completed implementation verification and French-to-English translation experiment.

## Implementation verification

- The verification suite recorded 16 passing tests in WSL, including interrupted-versus-uninterrupted training on CPU and BF16 CUDA.
- Accumulated gradients matched one equivalent batch with unequal target lengths and ignored padding.
- Optimizer/scheduler recovery, corrupted-checkpoint fallback, and completion recovery without an extra update passed.
- The notebook executed in observation mode with a WSL kernel.
- `pip check` found no broken requirements in the WSL environment.
- The small smoke pipeline completed with real Marian weights, CUDA, semantic scoring, and reporting.
- The normal profile completed all pipeline stages.
- An artifact review confirmed split isolation, token limits, paired prediction identities, exactly two observation examples, matching source hashes, and valid checkpoint checksums.
- Resuming the completed normal run skipped all stages, and the translation CLI loaded the selected checkpoint successfully.

## Experiment

The experiment used 20,000 training pairs, 500 validation pairs, and 1,000 test pairs. Each split contained equal numbers of Books and OPUS-100 examples.

Training ran for two epochs and 2,500 optimizer updates with a microbatch of two, eight accumulation steps, and BF16 mixed precision. The checkpoint with the highest validation chrF++ supplied the final test predictions.

## Translation quality

| Metric | Baseline | Fine-tuned | Difference |
|---|---:|---:|---:|
| BLEU | 30.2976 | 31.1971 | +0.8995 |
| chrF++ | 51.2762 | 52.2286 | +0.9524 |
| TER | 61.4147 | 60.1124 | -1.3023 |
| BERTScore F1 | 0.928958 | 0.931425 | +0.002467 |
| Token-weighted NLL | 2.1786 | 1.8671 | -0.3115 |
| Perplexity | 8.8338 | 6.4695 | -2.3643 |

The paired 95% difference intervals were:

- BLEU: +0.3227 to +1.4099.
- chrF++: +0.5646 to +1.3391.
- TER: -1.9598 to -0.6404.
- BERTScore F1: +0.0016 to +0.0033.

Books BLEU improved from 22.9808 to 24.9871, while OPUS-100 BLEU declined from 39.2752 to 38.8304. The aggregate improvement does not establish an improvement across all domains. Confidence intervals describe the fixed test sample and do not account for pretraining overlap or training-seed variation.

The two examples selected before training exposed source noise, including a misaligned OPUS-100 reference. Structural cleaning and length filters do not remove every alignment error. See the [evaluation protocol](evaluation.md) for metric definitions and limitations.

## Find results in a run directory

| File | Contents |
|---|---|
| `report.md` | Local summary, metrics, and examples |
| `comparison.csv` | Aggregate before-and-after quality scores |
| `confidence_intervals.json` | Paired confidence intervals |
| `examples.json` | The two examples selected before training |
| `training_result.json` | Training outcome and selected checkpoint location |

These paths are relative to the run directory printed by the CLI. Follow the [sharing guidance](workflow.md#sharing-results) when preparing a public copy.

The reported experiment used the corrected token-weighted loss. A regression test covers the Marian/Trainer gradient-accumulation mismatch found during development.
