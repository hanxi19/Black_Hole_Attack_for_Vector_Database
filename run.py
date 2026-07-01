#!/usr/bin/env python3
"""
Entry point: run the full black-hole attack pipeline and evaluate.

Usage:
    python run.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from process_data.data_manager import DataManager
from attack.pipeline import BlackHolePipeline
from evaluation.attack_evaluation import evaluate
from evaluation.recall_evaluation import evaluate_recall
from evaluation.detection_defense_evaluation import evaluate_defense
from evaluation.detection_defense_performance_loss_evaluation import evaluate_defense_performance_loss
from evaluation.mitigation_defense_evaluation import evaluate_mitigation_defense
from evaluation.mitigation_defense_performance_loss import evaluate_mitigation_performance_loss
from defense.detection_based import DetectionBasedDefense
from defense.mitigation_based import MitigationBasedDefense

# ═══════════════════════════════════════════════════════════════════════════════
#  Hyperparameters — modify here to experiment
# ═══════════════════════════════════════════════════════════════════════════════

MODEL = "contriever"
SRC_DATASET = "hotpotqa"

# Attack mode:
#   "default"  — train and attack the same dataset (victim = src)
#   "transfer" — train on src, inject into a different dataset
MODE = "default"
VICTIM_DATASET = "hotpotqa"  # only used when MODE == "transfer"

PREPROCESS_MODE = "default" # "default | query_trans"

CLUSTER_METHOD = "faiss_remote"  # "kmeans" | "minibatch_kmeans | adaptive | faiss_gpu | faiss_remote"
N_CLUSTERS: int | None = None  # None = auto: len(corpus)/1000 of target dataset
BATCH_SIZE = 30000
MAX_POINTS_PER_CENTROID: int | None = None  # None = use all data; set to N for FAISS minibatch-like sampling

NUM_COPIES = 10
EPSILON = 0.001
SEED = 42

INDEX_TYPE = "FlatIP"  # "FlatIP" | "IVF" | "HNSW" | "IVFPQ" (used for the poisoned index saved to disk)
INDEX_KWARGS: dict = {}  # e.g. {"nlist": 4096} for IVF, {"hnsw_M": 32} for HNSW

# EVAL_INDEX_TYPES = ["FlatIP", "IVF", "HNSW", "IVFPQ"]  # evaluate ALL these index types
EVAL_INDEX_TYPES = ["FlatIP"]

SAMPLE_QUERIES: int | None = 3000  # None = all; set to an integer to subsample
EVAL_K = 10

# Evaluation runners — add/remove functions to control which evals run.
# Detection-based:  evaluate_defense, evaluate_defense_performance_loss
# Mitigation-based: evaluate_mitigation_defense, evaluate_mitigation_performance_loss
EVAL_RUNNERS = [
    evaluate,
    # evaluate_recall,
    # evaluate_defense,
    # evaluate_defense_performance_loss,
    # evaluate_mitigation_defense,
    # evaluate_mitigation_performance_loss,
]

# Detection-based defense parameters
DEFENSE_N_CLUSTERS: int = 5000
DEFENSE_PROBE_RATIO: float = 0.005
DEFENSE_CLUSTER_METHOD: str = "faiss_gpu"
PROBE_K = 50

# Mitigation-based defense parameters
MITIGATION_METHODS: list[str] = ["cl2", "zn", "tcpr", "nohub"]
MITIGATION_TCPR_K: int = 10
MITIGATION_NOHUB_OUT_DIMS: int = 400
MITIGATION_NOHUB_N_ITER: int = 50
MITIGATION_NOHUB_MAX_SAMPLES: int = 2000

# ═══════════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent / "data"
VECTOR_DIR = ROOT / "vector"
DATASET_DIR = ROOT / "datasets"
OUTPUT_DIR: Path | None = None  # set in main() based on model+dataset to avoid parallel conflicts
RESULT_DIR = ROOT / "result"

RESULT_SUBDIR = ""  # subdirectory under RESULT_DIR, e.g. "main" or "ablation"


def _make_dm(dataset: str) -> DataManager:
    return DataManager(MODEL, dataset, vector_dir=str(VECTOR_DIR), dataset_dir=str(DATASET_DIR))


def run_pipeline() -> BlackHolePipeline:
    """Load source, optionally load victim (transfer mode), run attack, save, return result."""
    # Step 1: Load source
    print("=" * 60)
    print("  STEP 1: Load source data")
    print("=" * 60)
    source = _make_dm(SRC_DATASET)
    source.load_all()
    print(source.summarize())

    # Load victim if transfer mode
    if MODE == "transfer":
        print()
        print("=" * 60)
        print("  STEP 1b: Load victim data (transfer mode)")
        print("=" * 60)
        victim = _make_dm(VICTIM_DATASET)
        victim.load_all()
        print(victim.summarize())
    else:
        victim = None

    # Auto-compute n_clusters if not explicitly set
    n_clusters = N_CLUSTERS
    if n_clusters is None:
        target = victim if MODE == "transfer" and victim is not None else source
        n_clusters = max(1, len(target.corpus_texts) // 1000)
        print(f"  n_clusters auto: {n_clusters}  (corpus={len(target.corpus_texts)} // 1000)")

    # Step 2: Run attack pipeline
    print()
    print("=" * 60)
    print("  STEP 2: Run attack pipeline")
    print("=" * 60)
    pipeline = BlackHolePipeline(
        source,
        victim=victim,
        preprocess_mode=PREPROCESS_MODE,
        cluster_method=CLUSTER_METHOD,
        n_clusters=n_clusters,
        batch_size=BATCH_SIZE,
        max_points_per_centroid=MAX_POINTS_PER_CENTROID,
        num_copies=NUM_COPIES,
        epsilon=EPSILON,
        seed=SEED,
        index_type=INDEX_TYPE,
        **INDEX_KWARGS,
    )
    poisoned = pipeline.run()

    # Step 3: Save
    print()
    print("=" * 60)
    print("  STEP 3: Save poisoned data")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(OUTPUT_DIR))

    return pipeline


def run_defense(poisoned_dm: DataManager):
    """Run detection-based defense on both poisoned and clean corpora.

    Returns (clean_dm, defended_dm, defense, defended_clean_dm, defense_clean).
    Any of these may be None if the corresponding evaluation is not in EVAL_RUNNERS.
    """
    needs_defense = evaluate_defense in EVAL_RUNNERS
    needs_perf_loss = evaluate_defense_performance_loss in EVAL_RUNNERS

    if not needs_defense and not needs_perf_loss:
        return None, None, None, None, None

    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET

    print()
    print("=" * 60)
    print("  STEP 5: Detection-based defense")
    print("=" * 60)

    # Load clean DM (shared by both evaluations)
    clean_dm = DataManager(MODEL, eval_dataset, vector_dir=str(VECTOR_DIR), dataset_dir=str(DATASET_DIR))
    clean_dm.load_corpus()
    clean_dm.load_queries()
    print(clean_dm.summarize())
    print()

    defense = None
    defended_dm = None
    defense_clean = None
    defended_clean_dm = None

    # ── Defense on poisoned corpus ──────────────────────────────────
    if needs_defense:
        print("--- Defense on poisoned corpus ---")
        defense = DetectionBasedDefense(
            poisoned_dm,
            n_clusters=DEFENSE_N_CLUSTERS,
            probe_ratio=DEFENSE_PROBE_RATIO,
            k=PROBE_K,
            cluster_method=DEFENSE_CLUSTER_METHOD,
            batch_size=BATCH_SIZE,
        )
        defense.detect()
        defended_dm = defense.apply()
        defended_dm.query_vecs = poisoned_dm.query_vecs.copy()
        defended_dm.query_texts = poisoned_dm.query_texts.copy()
        print()

    # ── Defense on clean corpus (performance loss measurement) ──────
    if needs_perf_loss:
        print("--- Defense on clean corpus (performance loss) ---")
        defense_clean = DetectionBasedDefense(
            clean_dm,
            n_clusters=DEFENSE_N_CLUSTERS,
            probe_ratio=DEFENSE_PROBE_RATIO,
            k=PROBE_K,
            cluster_method=DEFENSE_CLUSTER_METHOD,
            batch_size=BATCH_SIZE,
        )
        defense_clean.detect()
        defended_clean_dm = defense_clean.apply()
        defended_clean_dm.query_vecs = clean_dm.query_vecs.copy()
        defended_clean_dm.query_texts = clean_dm.query_texts.copy()
        print()

    return clean_dm, defended_dm, defense, defended_clean_dm, defense_clean


def run_mitigation_defense(poisoned_dm: DataManager):
    """Run each mitigation method **independently** on poisoned and clean corpora.

    Returns (clean_dm, per_method), where ``per_method`` is a dict::

        {method: {"defended_dm": ..., "defense": ...,
                  "defended_clean_dm": ..., "defense_clean": ...}}
    """
    needs_mo = evaluate_mitigation_defense in EVAL_RUNNERS
    needs_perf = evaluate_mitigation_performance_loss in EVAL_RUNNERS

    if not needs_mo and not needs_perf:
        return None, {}

    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET

    print()
    print("=" * 60)
    print("  STEP 5b: Mitigation-based defense  (per-method)")
    print("=" * 60)

    # Load clean DM (shared across all methods)
    clean_dm = DataManager(MODEL, eval_dataset, vector_dir=str(VECTOR_DIR), dataset_dir=str(DATASET_DIR))
    clean_dm.load_corpus()
    clean_dm.load_queries()

    # ── Pre-sample queries so mitigation (especially TCPR) runs faster ──
    # TCPR is O(Q × N × d) — brute-force k-NN for every query against the
    # entire corpus.  Sampling queries *before* mitigation avoids wasted work
    # while producing identical per-query results (TCPR has no cross-query
    # dependency).  Evaluation runners receive sample=None thereafter because
    # queries are already trimmed.
    _n_q = clean_dm.query_vecs.shape[0]
    if SAMPLE_QUERIES is not None and SAMPLE_QUERIES < _n_q:
        _rng = np.random.default_rng(SEED)
        _sampled_idx = _rng.choice(_n_q, size=SAMPLE_QUERIES, replace=False)

        clean_dm.query_vecs = clean_dm.query_vecs[_sampled_idx]
        clean_dm.query_texts = clean_dm.query_texts.iloc[_sampled_idx].reset_index(drop=True)

        # Match the same indices for poisoned_dm (same queries, same seed)
        if poisoned_dm.query_vecs is not None and poisoned_dm.query_vecs.shape[0] == _n_q:
            poisoned_dm.query_vecs = poisoned_dm.query_vecs[_sampled_idx]
            poisoned_dm.query_texts = poisoned_dm.query_texts.iloc[_sampled_idx].reset_index(drop=True)

        print(f"  Pre-sampled queries: {SAMPLE_QUERIES}/{_n_q} (seed={SEED})")
        print(f"  TCPR complexity: {SAMPLE_QUERIES:,} × {clean_dm.corpus_vecs.shape[0]:,} → "
              f"~{SAMPLE_QUERIES * clean_dm.corpus_vecs.shape[0] / 1e9:.1f}B pairwise ops")

    print(clean_dm.summarize())
    print()

    per_method: dict = {}

    for method in MITIGATION_METHODS:
        print(f"--- Mitigation: {method} ---")
        entry: dict = {}

        if needs_mo:
            defense = MitigationBasedDefense(
                poisoned_dm, methods=[method],
                tcpr_k=MITIGATION_TCPR_K,
                nohub_out_dims=MITIGATION_NOHUB_OUT_DIMS,
                nohub_n_iter=MITIGATION_NOHUB_N_ITER,
                nohub_max_samples=MITIGATION_NOHUB_MAX_SAMPLES,
            )
            entry["defended_dm"] = defense.apply()
            entry["defense"] = defense

        if needs_perf:
            defense_clean = MitigationBasedDefense(
                clean_dm, methods=[method],
                tcpr_k=MITIGATION_TCPR_K,
                nohub_out_dims=MITIGATION_NOHUB_OUT_DIMS,
                nohub_n_iter=MITIGATION_NOHUB_N_ITER,
                nohub_max_samples=MITIGATION_NOHUB_MAX_SAMPLES,
            )
            entry["defended_clean_dm"] = defense_clean.apply()
            entry["defense_clean"] = defense_clean

        per_method[method] = entry
        print()

    return clean_dm, per_method


def run_evaluation(poisoned_dm: DataManager,
                   clean_dm: DataManager | None = None,
                   # Detection-based defense
                   det_defended_dm: DataManager | None = None,
                   det_defense: DetectionBasedDefense | None = None,
                   det_defended_clean_dm: DataManager | None = None,
                   det_defense_clean: DetectionBasedDefense | None = None,
                   # Mitigation-based defense  (per-method dicts + pre-sampled clean DM)
                   mit_per_method: dict | None = None,
                   clean_dm_mit: DataManager | None = None):
    """Run all evaluation functions in EVAL_RUNNERS and return their results."""
    results: dict = {}
    mit_per_method = mit_per_method or {}

    step = 6
    for runner in EVAL_RUNNERS:
        print()
        print("=" * 60)
        print(f"  STEP {step}: {runner.__name__}")
        print("=" * 60)

        if runner is evaluate:
            results["attack"] = runner(poisoned_dm, k=EVAL_K, sample=SAMPLE_QUERIES, index_types=EVAL_INDEX_TYPES)

        elif runner is evaluate_recall:
            results["recall"] = runner(poisoned_dm, k=EVAL_K, sample=SAMPLE_QUERIES, index_types=EVAL_INDEX_TYPES)

        elif runner is evaluate_defense:
            results["defense"] = runner(
                clean_dm, poisoned_dm, det_defended_dm,
                k=EVAL_K, sample=SAMPLE_QUERIES, index_types=EVAL_INDEX_TYPES,
                defense_timing=det_defense.timing if det_defense else None,
            )

        elif runner is evaluate_defense_performance_loss:
            results["defense_perf_loss"] = runner(
                clean_dm, det_defended_clean_dm,
                k=EVAL_K, sample=SAMPLE_QUERIES, index_types=EVAL_INDEX_TYPES,
                defense_timing=det_defense_clean.timing if det_defense_clean else None,
            )

        elif runner is evaluate_mitigation_defense:
            for method, entry in mit_per_method.items():
                key = f"mitigation_defense_{method}"
                results[key] = runner(
                    clean_dm_mit if clean_dm_mit is not None else clean_dm,
                    poisoned_dm, entry["defended_dm"],
                    k=EVAL_K, sample=None, index_types=EVAL_INDEX_TYPES,
                    defense_timing=entry["defense"].timing,
                )

        elif runner is evaluate_mitigation_performance_loss:
            for method, entry in mit_per_method.items():
                key = f"mitigation_perf_loss_{method}"
                results[key] = runner(
                    clean_dm_mit if clean_dm_mit is not None else clean_dm,
                    entry["defended_clean_dm"],
                    k=EVAL_K, sample=None, index_types=EVAL_INDEX_TYPES,
                    defense_timing=entry["defense_clean"].timing,
                )

        step += 1

    return results


def save_results(results: dict, cluster_time: float | None = None) -> None:
    """Persist all evaluation results to JSON."""
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET
    payload: dict = {
        "config": {
            "model": MODEL,
            "mode": MODE,
            "src_dataset": SRC_DATASET,
            "victim_dataset": eval_dataset,
            "preprocess_mode": PREPROCESS_MODE,
            "cluster_method": CLUSTER_METHOD,
            "n_clusters": N_CLUSTERS,
            "batch_size": BATCH_SIZE,
            "max_points_per_centroid": MAX_POINTS_PER_CENTROID,
            "num_copies": NUM_COPIES,
            "epsilon": EPSILON,
            "seed": SEED,
            "index_type": INDEX_TYPE,
            "eval_index_types": EVAL_INDEX_TYPES,
            "sample_queries": SAMPLE_QUERIES,
            "cluster_time_sec": cluster_time,
            "eval_runners": [r.__name__ for r in EVAL_RUNNERS],
        },
    }

    # Attack results
    if "attack" in results:
        attack_metrics = results["attack"]
        payload["attack"] = {
            idx_type: {
                f"MO@{EVAL_K}": m.mo_at_k,
                f"MO@{EVAL_K}_std": m.mo_at_k_std,
                "ASR": m.asr,
                "FPR_mean": m.fpr_mean,
                "FPR_std": m.fpr_std,
                "k": m.k,
                "num_queries": m.num_queries,
            }
            for idx_type, m in attack_metrics.items()
        }

    # Recall results
    if "recall" in results:
        recall_metrics = results["recall"]
        payload["recall"] = {
            idx_type: {
                f"Recall@{EVAL_K}_clean": r.clean.recall_at_k,
                f"Recall@{EVAL_K}_clean_std": r.clean.recall_at_k_std,
                f"Recall@{EVAL_K}_poisoned": r.poisoned.recall_at_k,
                f"Recall@{EVAL_K}_poisoned_std": r.poisoned.recall_at_k_std,
                "delta": r.delta,
                "k": r.clean.k,
                "num_queries": r.clean.num_queries,
            }
            for idx_type, r in recall_metrics.items()
        }

    # Defense results
    if "defense" in results:
        defense_metrics = results["defense"]
        payload["defense"] = {
            idx_type: {
                f"MO@{EVAL_K}_before": r.mo_before,
                f"MO@{EVAL_K}_after": r.mo_after,
                "n_clean": r.n_clean,
                "n_poisoned": r.n_poisoned,
                "n_defended": r.n_defended,
                "n_malicious_removed": r.n_malicious_removed,
                "n_benign_removed": r.n_benign_removed,
                "defense_cluster_time": r.defense_cluster_time,
                "defense_probe_search_time": r.defense_probe_search_time,
                "defense_total_time": r.defense_total_time,
            }
            for idx_type, r in defense_metrics.items()
        }

    # Defense performance loss results (defense on clean data)
    if "defense_perf_loss" in results:
        perf_loss_metrics = results["defense_perf_loss"]
        payload["defense_perf_loss"] = {
            idx_type: {
                f"R@{EVAL_K}": r.recall_at_k,
                f"R@{EVAL_K}_std": r.recall_at_k_std,
                "n_clean": r.n_clean,
                "n_defended_clean": r.n_defended_clean,
                "n_removed": r.n_removed,
                "defense_cluster_time": r.defense_cluster_time,
                "defense_probe_search_time": r.defense_probe_search_time,
                "defense_total_time": r.defense_total_time,
            }
            for idx_type, r in perf_loss_metrics.items()
        }

    # Mitigation defense results (per-method keys: mitigation_defense_cl2, ...)
    for key, mit_def in results.items():
        if key.startswith("mitigation_defense_"):
            method = key[len("mitigation_defense_"):]
            payload.setdefault("mitigation_defense", {})[method] = {
                idx_type: {
                    f"MO@{EVAL_K}_before": r.mo_before,
                    f"MO@{EVAL_K}_after": r.mo_after,
                    "n_clean": r.n_clean,
                    "n_poisoned": r.n_poisoned,
                    "n_defended": r.n_defended,
                    "n_malicious_total": r.n_malicious_total,
                    "defense_timing": r.defense_timing,
                }
                for idx_type, r in mit_def.items()
            }

    # Mitigation performance loss results (per-method keys)
    for key, mit_perf in results.items():
        if key.startswith("mitigation_perf_loss_"):
            method = key[len("mitigation_perf_loss_"):]
            payload.setdefault("mitigation_perf_loss", {})[method] = {
                idx_type: {
                    f"R@{EVAL_K}": r.recall_at_k,
                    f"R@{EVAL_K}_std": r.recall_at_k_std,
                    "n_clean": r.n_clean,
                    "n_defended_clean": r.n_defended_clean,
                    "n_removed": r.n_removed,
                    "defense_timing": r.defense_timing,
                }
                for idx_type, r in mit_perf.items()
            }

    result_dir = RESULT_DIR / RESULT_SUBDIR if RESULT_SUBDIR else RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{MODEL}_{eval_dataset}.json"
    result_path = result_dir / filename
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {result_path}")


def print_summary(results: dict) -> None:
    """Print final evaluation summary for all evals that were run."""
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET
    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Model:          {MODEL}")
    print(f"  Mode:           {MODE}")
    print(f"  Src dataset:    {SRC_DATASET}")
    print(f"  Victim dataset: {eval_dataset}")
    print(f"  Clusters:       {N_CLUSTERS}")
    print(f"  Copies:         {NUM_COPIES}")
    print(f"  Epsilon:        {EPSILON}")
    print(f"  Seed:           {SEED}")

    # -- Attack metrics --
    if "attack" in results:
        attack_metrics = results["attack"]
        print()
        print(f"  --- Attack Effectiveness ---")
        header = f"  {'Index':<8s}  {'MO@{EVAL_K}':>10s}  {'ASR':>8s}  {'FPR':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for idx_type, m in attack_metrics.items():
            print(f"  {idx_type:<8s}  {m.mo_at_k:>10.4f}  {m.asr:>8.4f}  {m.fpr_mean:>8.2f}")

    # -- Recall metrics --
    if "recall" in results:
        recall_metrics = results["recall"]
        print()
        print(f"  --- Recall@{EVAL_K} (clean vs poisoned) ---")
        header = f"  {'Index':<8s}  {'Clean':>10s}  {'Poisoned':>10s}  {'Delta':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for idx_type, r in recall_metrics.items():
            print(f"  {idx_type:<8s}  {r.clean.recall_at_k:>10.4f}  {r.poisoned.recall_at_k:>10.4f}  {r.delta:>+8.4f}")

    # -- Defense metrics --
    if "defense" in results:
        defense_metrics = results["defense"]
        print()
        print(f"  --- Detection-Based Defense ---")
        header = f"  {'Index':<8s}  {'MO@{EVAL_K} before':>15s}  {'MO@{EVAL_K} after':>14s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for idx_type, r in defense_metrics.items():
            print(f"  {idx_type:<8s}  {r.mo_before:>15.4f}  {r.mo_after:>14.4f}")

    # -- Defense performance loss (defense on clean data) --
    if "defense_perf_loss" in results:
        perf_loss_metrics = results["defense_perf_loss"]
        print()
        print(f"  --- Defense Performance Loss (on clean data) ---")
        header = f"  {'Index':<8s}  {'R@{EVAL_K}':>10s}  {'Removed':>8s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for idx_type, r in perf_loss_metrics.items():
            print(f"  {idx_type:<8s}  {r.recall_at_k:>10.4f}  {r.n_removed:>8d}")

    # -- Mitigation defense metrics (per-method) --
    mit_def_keys = sorted(k for k in results if k.startswith("mitigation_defense_"))
    if mit_def_keys:
        print()
        print(f"  --- Mitigation-Based Defense (per method) ---")
        header = f"  {'Method':<8s}  {'MO@{EVAL_K} before':>15s}  {'MO@{EVAL_K} after':>14s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in mit_def_keys:
            method = key[len("mitigation_defense_"):]
            for idx_type, r in results[key].items():
                print(f"  {method:<8s}  {r.mo_before:>15.4f}  {r.mo_after:>14.4f}")

    # -- Mitigation performance loss (per-method) --
    mit_perf_keys = sorted(k for k in results if k.startswith("mitigation_perf_loss_"))
    if mit_perf_keys:
        print()
        print(f"  --- Mitigation Performance Loss (per method, on clean data) ---")
        header = f"  {'Method':<8s}  {'R@{EVAL_K}':>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in mit_perf_keys:
            method = key[len("mitigation_perf_loss_"):]
            for idx_type, r in results[key].items():
                print(f"  {method:<8s}  {r.recall_at_k:>10.4f}")

    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Black-hole attack pipeline")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--src", "--src-dataset", dest="src_dataset", default=SRC_DATASET)
    parser.add_argument("--mode", default=MODE, choices=["default", "transfer"])
    parser.add_argument("--victim", "--victim-dataset", dest="victim_dataset", default=VICTIM_DATASET)
    parser.add_argument("--preprocess", dest="preprocess_mode", default=PREPROCESS_MODE)
    parser.add_argument("--cluster", dest="cluster_method", default=CLUSTER_METHOD)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-points-per-centroid", type=int, default=MAX_POINTS_PER_CENTROID,
                        help="FAISS points per centroid per iter (None=all data, N=minibatch-like)")
    parser.add_argument("--num-copies", type=int, default=NUM_COPIES)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--index-type", default=INDEX_TYPE)
    parser.add_argument("--sample-queries", type=int, default=SAMPLE_QUERIES,
                        help="Number of queries to sample for evaluation (default: all)")
    parser.add_argument("--eval-k", type=int, default=EVAL_K)
    parser.add_argument("--eval-index-types", nargs="+", default=EVAL_INDEX_TYPES,
                        choices=["FlatIP", "FlatL2", "IVF", "HNSW", "IVFPQ"],
                        help="ANN index types to evaluate (default: all)")
    parser.add_argument("--eval-runners", nargs="+",
                        default=["evaluate_defense", "evaluate_defense_performance_loss"],
                        choices=["evaluate", "evaluate_recall",
                                 "evaluate_defense", "evaluate_defense_performance_loss",
                                 "evaluate_mitigation_defense", "evaluate_mitigation_performance_loss"],
                        help="Which evaluation runners to activate")
    parser.add_argument("--result-subdir", default=RESULT_SUBDIR,
                        help="Subdirectory under data/result/ for organizing experiment runs")
    parser.add_argument("--output-dir", default=None,
                        help="Override OUTPUT_DIR (default: data/poisoned/{model}_{dataset})")
    parser.add_argument("--mitigation-methods", nargs="+", default=MITIGATION_METHODS,
                        choices=["cl2", "zn", "tcpr", "nohub"],
                        help="Mitigation methods to apply (default: all four)")
    parser.add_argument("--mitigation-tcpr-k", type=int, default=MITIGATION_TCPR_K)
    parser.add_argument("--mitigation-nohub-out-dims", type=int, default=MITIGATION_NOHUB_OUT_DIMS)
    parser.add_argument("--mitigation-nohub-n-iter", type=int, default=MITIGATION_NOHUB_N_ITER)
    parser.add_argument("--mitigation-nohub-max-samples", type=int, default=MITIGATION_NOHUB_MAX_SAMPLES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL = args.model
    SRC_DATASET = args.src_dataset
    MODE = args.mode
    VICTIM_DATASET = args.victim_dataset
    PREPROCESS_MODE = args.preprocess_mode
    CLUSTER_METHOD = args.cluster_method
    N_CLUSTERS = args.n_clusters
    BATCH_SIZE = args.batch_size
    MAX_POINTS_PER_CENTROID = args.max_points_per_centroid
    NUM_COPIES = args.num_copies
    EPSILON = args.epsilon
    SEED = args.seed
    INDEX_TYPE = args.index_type
    EVAL_INDEX_TYPES = args.eval_index_types
    SAMPLE_QUERIES = args.sample_queries
    EVAL_K = args.eval_k
    RESULT_SUBDIR = args.result_subdir
    _RUNNER_MAP = {
        "evaluate": evaluate,
        "evaluate_recall": evaluate_recall,
        "evaluate_defense": evaluate_defense,
        "evaluate_defense_performance_loss": evaluate_defense_performance_loss,
        "evaluate_mitigation_defense": evaluate_mitigation_defense,
        "evaluate_mitigation_performance_loss": evaluate_mitigation_performance_loss,
    }
    EVAL_RUNNERS = [_RUNNER_MAP[name] for name in args.eval_runners]
    MITIGATION_METHODS = args.mitigation_methods
    MITIGATION_TCPR_K = args.mitigation_tcpr_k
    MITIGATION_NOHUB_OUT_DIMS = args.mitigation_nohub_out_dims
    MITIGATION_NOHUB_N_ITER = args.mitigation_nohub_n_iter
    MITIGATION_NOHUB_MAX_SAMPLES = args.mitigation_nohub_max_samples
    OUTPUT_DIR = Path(args.output_dir) if args.output_dir else (ROOT / "poisoned" / f"{MODEL}_{SRC_DATASET}")

    pipeline = run_pipeline()

    # Load poisoned data for evaluation
    eval_dataset = VICTIM_DATASET if MODE == "transfer" else SRC_DATASET
    print()
    print("=" * 60)
    print("  STEP 4: Load poisoned data for evaluation")
    print("=" * 60)
    poisoned_dm = DataManager(MODEL, eval_dataset, vector_dir=str(OUTPUT_DIR), dataset_dir=str(DATASET_DIR))
    poisoned_dm.load_corpus()
    poisoned_dm.load_queries()
    try:
        poisoned_dm.load_index()
    except FileNotFoundError:
        print("  (no saved index found; will build each type on the fly)")
    print(poisoned_dm.summarize())

    # ── Run defenses and evaluations ────────────────────────────────
    # Detection-based defense
    clean_dm, det_defended, det_defense, det_defended_clean, det_defense_clean = run_defense(poisoned_dm)
    # Mitigation-based defense (per-method, independent)
    clean_dm_mit, mit_per_method = run_mitigation_defense(poisoned_dm)

    _clean_dm = clean_dm if clean_dm is not None else clean_dm_mit
    results = run_evaluation(
        poisoned_dm, _clean_dm,
        det_defended_dm=det_defended, det_defense=det_defense,
        det_defended_clean_dm=det_defended_clean, det_defense_clean=det_defense_clean,
        mit_per_method=mit_per_method,
        clean_dm_mit=clean_dm_mit,
    )
    save_results(results, pipeline.cluster_time)
    print_summary(results)
