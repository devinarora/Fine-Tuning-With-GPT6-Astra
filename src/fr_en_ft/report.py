"""Saved automatic comparisons and exactly two preregistered translation examples."""

import csv
from .evaluate import paired_bootstrap
from .state import atomic_json, read_json


def report(run):
    baseline = read_json(run.path / "baseline_metrics.json")
    final = read_json(run.path / "final_metrics.json")
    a = read_json(run.path / "baseline_predictions.json")
    b = read_json(run.path / "final_predictions.json")
    intervals = paired_bootstrap(a, b, run.config.evaluation.bootstrap_samples, run.config.seed)
    atomic_json(run.path / "confidence_intervals.json", intervals)
    rows = []
    for group in ("overall", "books", "opus"):
        before = baseline["overall"] if group == "overall" else baseline["by_source"][group]
        after = final["overall"] if group == "overall" else final["by_source"][group]
        for metric in ("bleu", "chrf++", "ter", "bertscore_precision", "bertscore_recall", "bertscore_f1", "nll", "perplexity"):
            if metric in before and before[metric] is not None and after[metric] is not None:
                rows.append({"group": group, "metric": metric, "baseline": before[metric], "fine_tuned": after[metric],
                             "delta": after[metric] - before[metric]})
    with (run.path / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "metric", "baseline", "fine_tuned", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    by_a, by_b = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    ids = read_json(run.path / "data_manifest.json")["example_ids"]
    examples = [{"id": key, "corpus": by_a[key]["corpus"], "french": by_a[key]["source"],
                 "reference": by_a[key]["target"], "baseline": by_a[key]["hypothesis"], "fine_tuned": by_b[key]["hypothesis"]}
                for key in ids]
    atomic_json(run.path / "examples.json", examples)
    figure_path = learning_curve(run)
    lines = ["# French-to-English experiment", "", f"Run: `{run.path.name}`", "",
             f"Active automated time: {run.elapsed / 60:.1f} minutes.",
             f"Original budget: {run.config.runtime.budget_seconds / 60:.1f} minutes.",
             f"Explicit budget extensions: {run.state.get('extra_seconds', 0) / 60:.1f} minutes.", "",
             "## Automatic evaluation", "", "| Source | Metric | Baseline | Fine-tuned | Difference |",
             "|---|---|---:|---:|---:|"]
    lines.extend(f"| {r['group']} | {r['metric']} | {r['baseline']:.4f} | {r['fine_tuned']:.4f} | {r['delta']:+.4f} |" for r in rows)
    lines += ["", "Lower TER, NLL, and perplexity are better. Higher BLEU, chrF++, and BERTScore are better.",
              "BLEU, chrF++, and TER use SacreBLEU's scale. BERTScore is unscaled precision/recall/F1.", "",
              "## Paired 95% confidence intervals", ""]
    for key, ci in intervals["intervals_95"].items():
        lines.append(f"- {key} difference: {ci['low']:+.4f} to {ci['high']:+.4f}.")
    lines += ["", "Intervals describe test-sample uncertainty, not variation across training seeds.",
              "## Two fixed examples", ""]
    for example in examples:
        lines += [f"### {example['corpus']}", ""]
        for key in ("french", "reference", "baseline", "fine_tuned"):
            lines += [f"{key.replace('_', ' ').capitalize()}:", "", example[key], ""]
    if figure_path:
        lines += ["## Learning curves", "", f"![Training and validation curves]({figure_path.name})", ""]
    lines += ["## Interpretation limits", "",
              "The base model may have seen OPUS material before this experiment. These are fine-tuning holdouts, not certified unseen pretraining data.",
              "Books lacks reliable document identifiers. OPUS-100 lacks reliable spoken-language labels.",
              "The report evaluates eligible short text after documented filtering. It does not establish document translation quality.",
              "The two examples are for observation. Human judgments play no role in training or evaluation.",
              "Full metrics, exclusions, signatures, decoding settings, timings, and dataset identities are saved alongside this report.", ""]
    (run.path / "report.md").write_text("\n".join(lines), encoding="utf-8")
    run.complete("report")
    run.log("report_complete", path=str(run.path / "report.md"))


def learning_curve(run):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    events = [json.loads(line) for line in (run.path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    losses = [r for r in events if r["event"] == "trainer" and "loss" in r]
    validation = [r for r in events if r["event"] == "trainer" and "eval_chrf" in r]
    if not losses and not validation:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["step"] for r in losses], [r["loss"] for r in losses])
    axes[0].set(xlabel="Optimizer update", ylabel="Training loss")
    axes[1].plot([r["step"] for r in validation], [r["eval_chrf"] for r in validation], marker="o")
    axes[1].set(xlabel="Optimizer update", ylabel="Validation chrF++")
    fig.tight_layout()
    path = run.path / "learning-curves.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
