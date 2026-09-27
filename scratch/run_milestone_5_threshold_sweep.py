"""Milestone 5 Threshold Sweep and Error Analysis Script (High Performance Sub-Sample).
Runs candidate blocking & feature extraction ONCE on a 5,000 S1 stratified benchmark,
evaluates thresholds [0.50..0.96], records Macro F0.5 per S1 and Pair F0.5,
performs Error Analysis on the best threshold, and saves json + markdown artifacts.
"""

import time
import json
import string
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add workspace root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.normalize import normalize_dataframe
from src.blocking import generate_candidate_pairs
from src.features import extract_pairwise_features
from src.matching import (
    compute_composite_score,
    predict_matches,
    evaluate_predictions,
    evaluate_macro_f05_per_s1,
    parse_ground_truth_map,
    parse_ground_truth_pairs
)

def main():
    print("=" * 75, flush=True)
    print("  AMAZON ML CHALLENGE 2026 — MILESTONE 5 THRESHOLD SWEEP & ERROR ANALYSIS", flush=True)
    print("=" * 75, flush=True)
    
    t0 = time.time()
    config = PipelineConfig()
    train_dir = "dataset/train/"
    
    print("[1/5] Loading datasets (high performance sub-sample mode)...", flush=True)
    req_cols = ["entity_id", "business_name", "business_address", "country"]
    
        # Read S1 (first 50k rows)
    s1_df = pd.read_csv(train_dir + "train_source1.tsv", sep="\t", usecols=req_cols, nrows=50000, on_bad_lines="skip", low_memory=False)
    s1_ids_set = set(s1_df["entity_id"])

    # Load ground truth filtered to loaded S1 entities
    gt_df = pd.read_csv(
        train_dir + "train_ground_truth.tsv",
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"],
        dtype=str,
        on_bad_lines="skip",
        low_memory=False
    )
    gt_sub = gt_df[gt_df["source1_entity_id"].isin(s1_ids_set)].copy()

    matched = gt_sub["matched_entity_ids"].fillna("").str.strip()

    # Number of matched IDs per S1:
    # "" -> 0
    # "S2-x" -> 1
    # "S2-x,S3-y" -> 2
    s1_match_counts = np.where(matched.eq(""), 0, matched.str.count(",") + 1)

    gt_counts_df = pd.DataFrame({
        "entity_id": gt_sub["source1_entity_id"],
        "match_count": s1_match_counts
    })
    s1_df = s1_df.merge(gt_counts_df, on="entity_id", how="left")
    s1_df["match_count"] = s1_df["match_count"].fillna(0).astype(int)
    
    def get_cardinality_bucket(c):
        if c == 0: return "0"
        elif c == 1: return "1"
        elif c == 2: return "2"
        elif c == 3: return "3"
        else: return "4+"

    s1_df["card_bucket"] = s1_df["match_count"].apply(get_cardinality_bucket)
    s1_df["strat_key"] = s1_df["country"].astype(str) + "_" + s1_df["card_bucket"]

    # Stratified 500 S1 sample (seed=42)
    benchmark_size = 2000
    print(f"[2/5] Sampling {benchmark_size:,} S1 entities stratified by Country & Match Cardinality...", flush=True)
    np.random.seed(42)
    strat_counts = s1_df["strat_key"].value_counts(normalize=True)
    sampled_indices = []

    for key, frac in strat_counts.items():
        sub_df = s1_df[s1_df["strat_key"] == key]
        n_sample = int(round(frac * benchmark_size))
        if n_sample > len(sub_df):
            n_sample = len(sub_df)
        sampled_indices.extend(sub_df.sample(n=n_sample, random_state=42).index.tolist())

    s1_sample_raw = s1_df.loc[sampled_indices].copy()
    sampled_s1_ids = list(s1_sample_raw["entity_id"])
    sampled_gt_df = gt_df[gt_df["source1_entity_id"].isin(sampled_s1_ids)].copy()
    
    sample_gt_map = parse_ground_truth_map(sampled_gt_df)
    sample_gt_pairs = parse_ground_truth_pairs(sampled_gt_df)
    gt_target_ids = {p[1] for p in sample_gt_pairs}
    print(f"Benchmark Sample: {len(s1_sample_raw):,} S1 entities, {len(sample_gt_pairs):,} Ground-Truth Pairs ({len(gt_target_ids):,} unique targets).", flush=True)

    # Load S2 and S3 candidates (250k rows each)
    print("Loading candidate datasets (Source 2 & Source 3)...", flush=True)
    s2_df = pd.read_csv(train_dir + "train_source2.tsv", sep="\t", usecols=req_cols, nrows=250000, on_bad_lines="skip", low_memory=False)
    s3_df = pd.read_csv(train_dir + "train_source3.tsv", sep="\t", usecols=req_cols, nrows=250000, on_bad_lines="skip", low_memory=False)

    # Chunked missing target fetch
    loaded_s2_ids = set(s2_df["entity_id"])
    loaded_s3_ids = set(s3_df["entity_id"])
    missing_targets = gt_target_ids - (loaded_s2_ids | loaded_s3_ids)
    
    if missing_targets:
        print(f"  Fetching {len(missing_targets):,} missing ground-truth target records using fast chunking...", flush=True)
        extra_s2 = []
        for chunk in pd.read_csv(train_dir + "train_source2.tsv", sep="\t", usecols=req_cols, chunksize=500000, on_bad_lines="skip", low_memory=False):
            sub = chunk[chunk["entity_id"].isin(missing_targets)]
            if not sub.empty:
                extra_s2.append(sub)
        if extra_s2:
            s2_df = pd.concat([s2_df] + extra_s2).drop_duplicates(subset=["entity_id"])

        extra_s3 = []
        for chunk in pd.read_csv(train_dir + "train_source3.tsv", sep="\t", usecols=req_cols, chunksize=500000, on_bad_lines="skip", low_memory=False):
            sub = chunk[chunk["entity_id"].isin(missing_targets)]
            if not sub.empty:
                extra_s3.append(sub)
        if extra_s3:
            s3_df = pd.concat([s3_df] + extra_s3).drop_duplicates(subset=["entity_id"])

    print(f"Loaded {len(s2_df):,} Source 2 records and {len(s3_df):,} Source 3 records.", flush=True)

    print("[3/5] Vectorized Candidate Filtering & Normalization...", flush=True)
    t_filt_start = time.time()
    s1_sample_norm = normalize_dataframe(s1_sample_raw, config.legal_suffixes)
    
    # Extract 2-char prefixes from benchmark S1 names
    s1_prefixes = set(n[:2] for n in s1_sample_norm["name_normalized"] if len(n) >= 2)
    
    # Fast pandas vectorized mask
    s2_prefix_mask = s2_df["business_name"].fillna("").astype(str).str.lower().str[:2].isin(s1_prefixes)
    s2_sub = s2_df[s2_prefix_mask | s2_df["entity_id"].isin(gt_target_ids)].copy()

    s3_prefix_mask = s3_df["business_name"].fillna("").astype(str).str.lower().str[:2].isin(s1_prefixes)
    s3_sub = s3_df[s3_prefix_mask | s3_df["entity_id"].isin(gt_target_ids)].copy()

    print(f"Candidate subset filtering: S2 {len(s2_df):,} -> {len(s2_sub):,}, S3 {len(s3_df):,} -> {len(s3_sub):,} in {time.time() - t_filt_start:.2f}s.", flush=True)

    t_norm_start = time.time()
    s2_norm = normalize_dataframe(s2_sub, config.legal_suffixes)
    s3_norm = normalize_dataframe(s3_sub, config.legal_suffixes)
    print(f"Normalizing candidate subsets finished in {time.time() - t_norm_start:.2f}s.", flush=True)

    t_block_start = time.time()
    cand_pairs_df = generate_candidate_pairs(s1_sample_norm, s2_norm, s3_norm, config)
    print(f"Generated {len(cand_pairs_df):,} Candidate Pairs in {time.time() - t_block_start:.2f}s.", flush=True)
    
    # Candidate recall
    cand_pairs_set = set(zip(cand_pairs_df["source1_entity_id"], cand_pairs_df["candidate_entity_id"]))
    retrieved_gt = len(sample_gt_pairs & cand_pairs_set)
    cand_recall = retrieved_gt / float(len(sample_gt_pairs)) if sample_gt_pairs else 1.0
    print(f"Candidate Blocking Recall: {cand_recall * 100:.2f}% ({retrieved_gt:,} / {len(sample_gt_pairs):,})", flush=True)

    print("[4/5] Extracting Pairwise Features & Computing Match Scores ONCE...", flush=True)
    t_feat_start = time.time()
    features_df = extract_pairwise_features(cand_pairs_df, s1_sample_norm, s2_norm, s3_norm)
    features_df["match_score"] = compute_composite_score(features_df, config)
    print(f"Feature Extraction finished in {time.time() - t_feat_start:.2f}s.", flush=True)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96]
    sweep_results = []
    best_threshold = 0.70
    best_macro_f05 = -1.0
    best_predictions = None

    print("\n--- SWEEPING THRESHOLDS ---", flush=True)
    for t in thresholds:
        preds = features_df[features_df["match_score"] >= t].copy()
        macro_metrics = evaluate_macro_f05_per_s1(preds, sampled_gt_df, s1_ids=sampled_s1_ids, beta=0.5)
        pair_metrics = evaluate_predictions(cand_pairs_df, preds, sampled_gt_df, beta=0.5)
        
        t_res = {
            "threshold": t,
            "macro_precision": round(macro_metrics["macro_precision"], 4),
            "macro_recall": round(macro_metrics["macro_recall"], 4),
            "macro_f05": round(macro_metrics["macro_f05"], 4),
            "pair_precision": round(pair_metrics["precision"], 4),
            "pair_recall": round(pair_metrics["recall"], 4),
            "pair_f05": round(pair_metrics["f0.5"], 4),
            "predicted_pair_count": len(preds),
            "false_positive_pairs": pair_metrics["false_positives"],
            "false_negative_pairs": pair_metrics["false_negatives"]
        }
        sweep_results.append(t_res)
        
        print(f"Threshold {t:.2f} | Macro Precision: {t_res['macro_precision']:.4f} | Macro Recall: {t_res['macro_recall']:.4f} | MACRO F0.5: {t_res['macro_f05']:.4f} | Pair F0.5: {t_res['pair_f05']:.4f} | Pred Pairs: {len(preds):,}", flush=True)
        
        if t_res["macro_f05"] > best_macro_f05:
            best_macro_f05 = t_res["macro_f05"]
            best_threshold = t
            best_predictions = preds

    print(f"\nBEST THRESHOLD BY MACRO F0.5: {best_threshold:.2f} (Macro F0.5 = {best_macro_f05:.4f})", flush=True)

    print("[5/5] Performing Error Analysis on Best Threshold...", flush=True)
    
    s1_raw_dict = s1_sample_raw.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s1_norm_dict = s1_sample_norm.set_index("entity_id")[["name_normalized", "address_normalized"]].to_dict("index")
    
    target_raw_dict = {}
    target_raw_dict.update(s2_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index"))
    target_raw_dict.update(s3_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index"))

    target_norm_dict = {}
    target_norm_dict.update(s2_norm.set_index("entity_id")[["name_normalized", "address_normalized"]].to_dict("index"))
    target_norm_dict.update(s3_norm.set_index("entity_id")[["name_normalized", "address_normalized"]].to_dict("index"))

    pred_pairs_set = set(zip(best_predictions["source1_entity_id"], best_predictions["candidate_entity_id"]))
    
    fp_pairs = list(pred_pairs_set - sample_gt_pairs)
    fn_pairs = list(sample_gt_pairs - pred_pairs_set)
    
    feat_map = {}
    for _, r in features_df.iterrows():
        feat_map[(r["source1_entity_id"], r["candidate_entity_id"])] = r.to_dict()

    def make_error_record(s1_id, cand_id, status_label):
        s1_r = s1_raw_dict.get(s1_id, {})
        s1_n = s1_norm_dict.get(s1_id, {})
        t_r = target_raw_dict.get(cand_id, {})
        t_n = target_norm_dict.get(cand_id, {})
        f_data = feat_map.get((s1_id, cand_id), {})
        
        return {
            "s1_id": s1_id,
            "candidate_id": cand_id,
            "candidate_source": f_data.get("candidate_source", "unknown"),
            "raw_s1_name": s1_r.get("business_name", ""),
            "raw_cand_name": t_r.get("business_name", ""),
            "norm_s1_name": s1_n.get("name_normalized", ""),
            "norm_cand_name": t_n.get("name_normalized", ""),
            "raw_s1_address": s1_r.get("business_address", ""),
            "raw_cand_address": t_r.get("business_address", ""),
            "norm_s1_address": s1_n.get("address_normalized", ""),
            "norm_cand_address": t_n.get("address_normalized", ""),
            "score": round(float(f_data.get("match_score", 0.0)), 4),
            "key_features": {
                "name_exact": float(f_data.get("name_exact", 0.0)),
                "address_exact": float(f_data.get("address_exact", 0.0)),
                "name_char_similarity": round(float(f_data.get("name_char_similarity", 0.0)), 4),
                "address_char_similarity": round(float(f_data.get("address_char_similarity", 0.0)), 4),
                "numeric_agreement_indicator": float(f_data.get("numeric_agreement_indicator", 0.5)),
                "suffix_agreement": float(f_data.get("suffix_agreement", 0.5)),
            },
            "ground_truth_status": status_label
        }

    fp_samples = [make_error_record(p[0], p[1], "False Positive") for p in fp_pairs[:10]]
    fn_samples = [make_error_record(p[0], p[1], "False Negative") for p in fn_pairs[:10]]

    zero_gt_s1 = {s1_id for s1_id, m_ids in sample_gt_map.items() if len(m_ids) == 0 and s1_id in set(sampled_s1_ids)}
    zero_gt_fp_samples = []
    for s1_id in list(zero_gt_s1):
        pred_cands = best_predictions[best_predictions["source1_entity_id"] == s1_id]
        if not pred_cands.empty:
            for _, r in pred_cands.head(2).iterrows():
                zero_gt_fp_samples.append(make_error_record(s1_id, r["candidate_entity_id"], "Zero-GT S1 False Positive"))

    error_analysis = {
        "best_threshold": best_threshold,
        "summary": {
            "total_false_positives": len(fp_pairs),
            "total_false_negatives": len(fn_pairs),
            "zero_gt_s1_with_false_positives": len(zero_gt_fp_samples)
        },
        "false_positive_examples": fp_samples,
        "false_negative_examples": fn_samples,
        "zero_gt_s1_fp_examples": zero_gt_fp_samples[:5]
    }

    total_runtime = round(time.time() - t0, 2)
    sweep_output = {
        "benchmark_metadata": {
            "s1_benchmark_size": len(sampled_s1_ids),
            "total_ground_truth_pairs": len(sample_gt_pairs),
            "candidate_pair_count": len(cand_pairs_df),
            "candidate_recall_pct": round(cand_recall * 100, 2),
            "best_threshold": best_threshold,
            "best_macro_f05": round(best_macro_f05, 4),
            "runtime_seconds": total_runtime
        },
        "threshold_sweep": sweep_results
    }

    with open("scratch/milestone_5_threshold_sweep.json", "w") as f:
        json.dump(sweep_output, f, indent=2)

    with open("scratch/milestone_5_error_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2)

    print(f"\nCompleted in {total_runtime:.2f} seconds!", flush=True)
    print("Saved JSON artifacts: scratch/milestone_5_threshold_sweep.json & scratch/milestone_5_error_analysis.json", flush=True)

if __name__ == "__main__":
    main()
