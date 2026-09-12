"""Generated translation metrics, token-weighted loss, and resumable prediction batches."""

from collections import Counter
from contextlib import nullcontext
import gc
import math
import re
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from sacrebleu.metrics import BLEU, CHRF, TER

from .assets import asset_paths
from .data import prepared
from .environment import setup
from .model import collator, generation_options, load_model, load_tokenizer, token_features
from .state import atomic_json, read_json


class BudgetExhausted(RuntimeError):
    pass


def autocast(device):
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def metric_instances():
    return {"bleu": BLEU(tokenize="13a"), "chrf++": CHRF(word_order=2), "ter": TER()}


def lexical_scores(predictions, references):
    metrics = metric_instances()
    scores = {name: metric.corpus_score(predictions, [references]).score for name, metric in metrics.items()}
    return scores, {name: str(metric.get_signature()) for name, metric in metrics.items()}


def diagnose(source, reference, hypothesis):
    words = hypothesis.split()
    triples = Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    numbers = lambda text: Counter(re.findall(r"\d+(?:[.,]\d+)?", text))
    return {"empty": not hypothesis.strip(), "copied_source": hypothesis == source and source != reference,
            "repeated_trigram": max(triples.values(), default=0) >= 3,
            "number_mismatch": numbers(reference) != numbers(hypothesis),
            "length_ratio": len(words) / max(1, len(reference.split()))}


def predict_batch(model, tokenizer, rows, config, device):
    inputs = collator(tokenizer, model)([token_features(row) for row in rows])
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode(), autocast(device):
        outputs = model(**inputs, use_cache=False)
        labels = inputs["labels"]
        losses = F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                                 labels.reshape(-1), ignore_index=-100, reduction="none").reshape(labels.shape)
        sums = losses.sum(dim=1).cpu().tolist()
        counts = (labels != -100).sum(dim=1).cpu().tolist()
        del outputs, losses
        generated = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                   **generation_options(config))
    hypotheses = tokenizer.batch_decode(generated, skip_special_tokens=True)
    result = []
    for index, row in enumerate(rows):
        ids = generated[index].tolist()[1:]
        while ids and ids[-1] == tokenizer.pad_token_id:
            ids.pop()
        hypothesis = " ".join(hypotheses[index].split())
        result.append({key: row[key] for key in ("id", "corpus", "source", "target")})
        result[-1].update(hypothesis=hypothesis, source_tokens=len(row["input_ids"]),
                          target_tokens=counts[index], nll_sum=sums[index],
                          output_limit_hit=len(ids) >= config.data.max_target_tokens,
                          **diagnose(row["source"], row["target"], hypothesis))
    return result


def add_semantic_scores(records, scorer_path, config, device):
    from bert_score import BERTScorer
    # Explicit layer count is necessary for a revision-pinned local model path.
    scorer = BERTScorer(model_type=scorer_path, num_layers=config.evaluation.bertscore_layers,
                       device=device, batch_size=config.evaluation.batch_size, idf=False,
                       rescale_with_baseline=False, use_fast_tokenizer=False)
    maximum = min(scorer._tokenizer.model_max_length, 512)
    for record in records:
        for field in ("target", "hypothesis"):
            if len(scorer._tokenizer.encode(record[field], add_special_tokens=True)) > maximum:
                raise ValueError(f"BERTScore would truncate {field} for {record['id']}")
    with torch.inference_mode():
        precision, recall, f1 = scorer.score([r["hypothesis"] for r in records], [r["target"] for r in records])
    for row, p, r, f in zip(records, precision.tolist(), recall.tolist(), f1.tolist()):
        row.update(bertscore_precision=p, bertscore_recall=r, bertscore_f1=f)
    del scorer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def summarize(records):
    scores, signatures = lexical_scores([r["hypothesis"] for r in records], [r["target"] for r in records])
    token_count = sum(row["target_tokens"] for row in records)
    nll = sum(row["nll_sum"] for row in records) / max(1, token_count)
    scores.update(nll=nll, perplexity=math.exp(nll) if nll < 700 else None, examples=len(records), target_tokens=token_count)
    for field in ("bertscore_precision", "bertscore_recall", "bertscore_f1"):
        if field in records[0]:
            scores[field] = float(np.mean([r[field] for r in records]))
    scores["diagnostics"] = {key: sum(bool(row[key]) for row in records) for key in (
        "empty", "copied_source", "repeated_trigram", "number_mismatch", "output_limit_hit")}
    scores["diagnostics"]["mean_length_ratio"] = float(np.mean([r["length_ratio"] for r in records]))
    return scores, signatures


def grouped_scores(records):
    overall, signatures = summarize(records)
    result = {"overall": overall, "by_source": {}, "by_source_length": {}, "signatures": signatures}
    for corpus in sorted({r["corpus"] for r in records}):
        subset = [r for r in records if r["corpus"] == corpus]
        result["by_source"][corpus] = summarize(subset)[0]
    for low, high in ((1, 32), (33, 64), (65, 128), (129, 256), (257, 512)):
        subset = [r for r in records if low <= r["source_tokens"] <= high]
        if subset:
            result["by_source_length"][f"{low}-{high}"] = summarize(subset)[0]
    return result


def evaluate(run, kind, model_path):
    config = run.config
    device = setup(config)
    rows = list(prepared(run)["test"])
    paths = asset_paths(run)
    tokenizer = load_tokenizer(paths["model"])
    start = time.monotonic()
    model = load_model(model_path, device).eval()
    load_seconds = time.monotonic() - start
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    directory = run.path / "predictions" / kind
    directory.mkdir(parents=True, exist_ok=True)
    records, timings, latencies = [], [], []
    max_ram = psutil.Process().memory_info().rss
    for offset in range(0, len(rows), config.evaluation.batch_size):
        batch = rows[offset:offset + config.evaluation.batch_size]
        part = directory / f"part-{offset:08d}.json"
        if part.exists():
            saved = read_json(part)
            if saved["model_path"] != str(model_path) or [r["id"] for r in saved["records"]] != [r["id"] for r in batch]:
                raise ValueError("Cached predictions belong to a different model or test set")
        else:
            if run.remaining <= 0:
                raise BudgetExhausted("Evaluation budget exhausted; complete prediction batches are saved")
            synchronize(device)
            start = time.monotonic()
            result = predict_batch(model, tokenizer, batch, config, device)
            synchronize(device)
            saved = {"records": result, "seconds": time.monotonic() - start, "model_path": str(model_path)}
            atomic_json(part, saved)
            run.heartbeat()
        records.extend(saved["records"])
        timings.append(saved["seconds"])
        latencies.append(saved["seconds"] / len(batch))
        max_ram = max(max_ram, psutil.Process().memory_info().rss)
        if offset % max(config.evaluation.batch_size, 100) == 0:
            run.log("evaluation_progress", model=kind, examples=len(records), total=len(rows))
    peak_gpu = torch.cuda.max_memory_reserved() / 1024**3 if device == "cuda" else 0
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    semantic_start = time.monotonic()
    semantic_path = directory / "semantic.json"
    if config.evaluation.bertscore:
        if semantic_path.exists():
            semantic = read_json(semantic_path)
            if [r["id"] for r in semantic] != [r["id"] for r in records]:
                raise ValueError("Semantic results do not match the prediction order")
            records = semantic
        else:
            if run.remaining <= 0:
                raise BudgetExhausted("Budget exhausted before BERTScore; predictions are saved")
            add_semantic_scores(records, paths["scorer"], config, device)
            atomic_json(semantic_path, records)
    semantic_seconds = time.monotonic() - semantic_start
    result = grouped_scores(records)
    result["performance"] = {"generation_and_nll_seconds": sum(timings), "model_load_seconds": load_seconds,
                             "semantic_seconds": semantic_seconds, "examples_per_second": len(records) / max(sum(timings), 1e-9),
                             "amortized_latency_p50_seconds": float(np.percentile(latencies, 50)),
                             "amortized_latency_p95_seconds": float(np.percentile(latencies, 95)),
                             "generation_peak_reserved_gpu_gib": peak_gpu, "sampled_peak_process_ram_gib": max_ram / 1024**3,
                             "latency_scope": "Teacher-forced NLL plus beam generation, divided by batch size; not single-request latency"}
    result["bertscore_settings"] = {"enabled": config.evaluation.bertscore, "model": config.assets.scorer,
                                    "revision": config.assets.scorer_revision, "layers": config.evaluation.bertscore_layers,
                                    "idf": False, "rescale_with_baseline": False}
    atomic_json(run.path / f"{kind}_predictions.json", records)
    atomic_json(run.path / f"{kind}_metrics.json", result)
    run.complete(kind)
    run.log("evaluation_complete", model=kind, scores=result["overall"])
    return result


def paired_bootstrap(baseline, final, samples, seed):
    if [r["id"] for r in baseline] != [r["id"] for r in final]:
        raise ValueError("Paired evaluation requires identical ordered example IDs")
    refs = [r["target"] for r in baseline]
    if refs != [r["target"] for r in final]:
        raise ValueError("References differ between models")
    rng = np.random.default_rng(seed)
    groups = [np.array([i for i, row in enumerate(baseline) if row["corpus"] == corpus])
              for corpus in sorted({r["corpus"] for r in baseline})]
    metrics = metric_instances()
    statistics = {}
    for name, metric in metrics.items():
        # SacreBLEU's sufficient statistics avoid re-tokenizing every bootstrap replicate.
        statistics[name] = [np.asarray(metric._extract_corpus_statistics([r["hypothesis"] for r in rows], [refs]))
                            for rows in (baseline, final)]
    distributions = {name: [] for name in metrics}
    semantic = "bertscore_f1" in baseline[0]
    if semantic:
        distributions["bertscore_f1"] = []
        semantic_delta = np.array([b["bertscore_f1"] - a["bertscore_f1"] for a, b in zip(baseline, final)])
    for _ in range(samples):
        selected = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        for name, metric in metrics.items():
            a, b = statistics[name]
            delta = metric._compute_score_from_stats(b[selected].sum(axis=0).tolist()).score - metric._compute_score_from_stats(a[selected].sum(axis=0).tolist()).score
            distributions[name].append(delta)
        if semantic:
            distributions["bertscore_f1"].append(float(semantic_delta[selected].mean()))
    return {"method": "paired percentile bootstrap, stratified by source; test-sample uncertainty only",
            "samples": samples, "seed": seed, "delta_direction": "fine-tuned minus baseline; TER improvement is negative",
            "intervals_95": {key: {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}
                             for key, values in distributions.items()}}
