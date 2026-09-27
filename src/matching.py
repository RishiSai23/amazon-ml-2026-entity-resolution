"""Multi-signal deterministic matching model and metric evaluation functions."""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from src.config import PipelineConfig

def _get_series(df: pd.DataFrame, col: str, default_val: float = 0.0) -> pd.Series:
    """Safely extract a Series column from DataFrame or return default-filled Series."""
    if col in df.columns:
        return df[col]
    return pd.Series(default_val, index=df.index)


def compute_composite_score(df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    """Compute transparent multi-signal score for candidate pairs.
    
    Distinguishes:
      A) Strong evidence: Exact name, exact address, strong name + strong address,
         matching numeric address tokens, high core name token agreement.
      B) Medium evidence: High name similarity + moderate address similarity,
         moderate name + exact/near-exact address.
      C) Weak/Penalized evidence: Name similarity alone, conflicting suffixes,
         conflicting numeric house numbers.
         
    Args:
        df: Pairwise feature DataFrame.
        config: PipelineConfig containing weights & penalties.
        
    Returns:
        pd.Series of scalar scores bounded in [0.0, 1.0].
    """
    if df.empty:
        return pd.Series(dtype=float)
        
    # Name and Address feature extraction
    addr_char_s = _get_series(df, "address_char_similarity", 0.0)
    name_char_s = _get_series(df, "name_char_similarity", 0.0)
    name_sort_s = _get_series(df, "name_token_sort_similarity", 0.0)
    name_sim_effective = np.maximum(name_char_s, name_sort_s)

    # Name signal component
    name_sim = (
        config.w_name_exact * _get_series(df, "name_exact", 0.0) +
        config.w_name_char * name_sim_effective +
        config.w_core_jaccard * _get_series(df, "core_name_jaccard", 0.0)
    )
    
    # Address signal component
    addr_sim = (
        config.w_address_exact * _get_series(df, "address_exact", 0.0) +
        config.w_address_char * _get_series(df, "address_char_similarity", 0.0) +
        config.w_address_jaccard * _get_series(df, "address_jaccard", 0.0)
    )
    
    # Auxiliary signals
    num_signal = _get_series(df, "numeric_agreement_indicator", 0.5)
    suffix_signal = _get_series(df, "suffix_agreement", 0.5)
    country_signal = _get_series(df, "country_match", 1.0)
    
    # Base weighted raw score
    raw_score = (
        0.45 * name_sim +
        0.35 * addr_sim +
        0.10 * num_signal +
        0.05 * suffix_signal +
        0.05 * country_signal
    )
    
    # Strong Evidence Boosts
    exact_name = _get_series(df, "name_exact", 0.0) == 1.0
    exact_addr = _get_series(df, "address_exact", 0.0) == 1.0
    high_name_char = name_sim_effective > 0.85
    high_addr_char = addr_char_s > 0.80
    high_addr = high_addr_char
    
    both_exact_boost = (exact_name & exact_addr).astype(float) * 0.20
    name_exact_high_addr_boost = (exact_name & (addr_char_s > 0.50)).astype(float) * 0.10
    addr_exact_high_name_boost = (exact_addr & (name_sim_effective > 0.60)).astype(float) * 0.10
    strong_both_boost = (high_name_char & high_addr_char).astype(float) * 0.10
    strong_address_boost = (high_addr & (num_signal >= 0.5)).astype(float) * 0.12
    
    # Weak Evidence Penalties
    num_mismatch_penalty = (
        (num_signal == 0.0) & (addr_char_s < 0.55)
    ).astype(float) * config.penalty_numeric_mismatch
    suffix_mismatch_penalty = (suffix_signal == 0.0).astype(float) * config.penalty_suffix_mismatch
    
    # Strong contradiction:
    # Exact normalized names are not sufficient when addresses have
    # strong evidence of being different.
    address_conflict_penalty = (
        exact_name
        & (addr_char_s < 0.50)
        & (num_signal == 0.0)
    ).astype(float) * -0.15
    
    total_score = (
        raw_score +
        both_exact_boost +
        name_exact_high_addr_boost +
        addr_exact_high_name_boost +
        strong_both_boost +
        strong_address_boost +
        num_mismatch_penalty +
        suffix_mismatch_penalty +
        address_conflict_penalty
    )
    
    return total_score.clip(0.0, 1.0)

def predict_matches(features_df: pd.DataFrame, config: PipelineConfig, threshold: float = None) -> pd.DataFrame:
    """Filter candidate pairs by composite similarity threshold.
    
    Args:
        features_df: Pairwise feature DataFrame.
        config: PipelineConfig.
        threshold: Optional threshold override.
        
    Returns:
        Filtered DataFrame containing matches exceeding threshold.
    """
    if features_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "match_score"])
        
    cutoff = threshold if threshold is not None else config.match_threshold
    
    df = features_df.copy()
    if "match_score" not in df.columns:
        df["match_score"] = compute_composite_score(df, config)
        
    predictions = df[df["match_score"] >= cutoff].copy()
    
    return predictions[[
        "source1_entity_id", "candidate_entity_id", "candidate_source", "match_score", "blocking_rules"
    ]]

def parse_ground_truth_map(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse ground truth DataFrame into dict mapping s1_id -> set of matched_entity_ids."""
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1_id = row.get("source1_entity_id", row.get("entity_id"))
        matched_str = str(row.get("matched_entity_ids", "")).strip()
        if matched_str and matched_str != "nan":
            matched_ids = {m.strip() for m in matched_str.split(",") if m.strip()}
            gt_map[s1_id] = matched_ids
        else:
            gt_map[s1_id] = set()
    return gt_map

def parse_ground_truth_pairs(gt_df: pd.DataFrame) -> Set[Tuple[str, str]]:
    """Parse ground truth DataFrame into set of (source1_id, matched_id) tuples."""
    gt_pairs = set()
    gt_map = parse_ground_truth_map(gt_df)
    for s1_id, m_ids in gt_map.items():
        for m_id in m_ids:
            gt_pairs.add((s1_id, m_id))
    return gt_pairs

def calculate_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F_beta metric."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

def evaluate_predictions(
    candidate_pairs_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    beta: float = 0.5
) -> Dict[str, float]:
    """Evaluate pooled pair-level performance (Precision, Recall, F0.5) for diagnostic comparison."""
    if gt_df.empty:
        return {"candidate_recall": 0.0, "precision": 0.0, "recall": 0.0, f"f{beta}": 0.0}
        
    gt_pairs = parse_ground_truth_pairs(gt_df)
    total_true_pairs = len(gt_pairs)
    
    if total_true_pairs == 0:
        return {"candidate_recall": 1.0, "precision": 1.0, "recall": 1.0, f"f{beta}": 1.0}
        
    cand_pairs = set(zip(candidate_pairs_df["source1_entity_id"], candidate_pairs_df["candidate_entity_id"])) if not candidate_pairs_df.empty else set()
    retrieved_true_in_candidates = len(gt_pairs & cand_pairs)
    candidate_recall = retrieved_true_in_candidates / float(total_true_pairs)
    
    pred_pairs = set(zip(predictions_df["source1_entity_id"], predictions_df["candidate_entity_id"])) if not predictions_df.empty else set()
    true_positives = len(gt_pairs & pred_pairs)
    false_positives = len(pred_pairs - gt_pairs)
    false_negatives = len(gt_pairs - pred_pairs)
    
    precision = true_positives / float(len(pred_pairs)) if pred_pairs else 1.0
    recall = true_positives / float(total_true_pairs)
    f_beta = calculate_f_beta(precision, recall, beta)
    
    return {
        "candidate_count": len(cand_pairs),
        "candidate_recall": candidate_recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        f"f{beta}": f_beta
    }

def evaluate_macro_f05_per_s1(
    predictions_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    s1_ids: List[str] = None,
    beta: float = 0.5
) -> Dict[str, float]:
    """Evaluate competition-style per-S1 macro-averaged precision, recall, and F0.5.
    
    For every S1 entity:
      - Precision = |pred & true| / |pred| (1.0 if empty pred & empty true; 0.0 if non-empty pred & empty true)
      - Recall = |pred & true| / |true| (1.0 if empty pred & empty true; 0.0 if empty pred & non-empty true)
      - F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    gt_map = parse_ground_truth_map(gt_df)
    
    if s1_ids is None:
        s1_ids = list(gt_map.keys())
        
    pred_map = {}
    if not predictions_df.empty:
        for s1_id, group in predictions_df.groupby("source1_entity_id"):
            pred_map[s1_id] = set(group["candidate_entity_id"])
            
    p_list = []
    r_list = []
    f_list = []
    
    zero_gt_count = 0
    one_gt_count = 0
    multi_gt_count = 0
    
    for s1_id in s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())
        
        n_true = len(true_set)
        if n_true == 0:
            zero_gt_count += 1
        elif n_true == 1:
            one_gt_count += 1
        else:
            multi_gt_count += 1
            
        if n_true == 0:
            if len(pred_set) == 0:
                p = 1.0
                r = 1.0
                f = 1.0
            else:
                p = 0.0
                r = 0.0
                f = 0.0
        else:
            if len(pred_set) == 0:
                p = 0.0
                r = 0.0
                f = 0.0
            else:
                tp = len(pred_set & true_set)
                p = tp / float(len(pred_set))
                r = tp / float(n_true)
                f = calculate_f_beta(p, r, beta)
                
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        
    macro_precision = float(np.mean(p_list)) if p_list else 0.0
    macro_recall = float(np.mean(r_list)) if r_list else 0.0
    macro_f05 = float(np.mean(f_list)) if f_list else 0.0
    
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f05": macro_f05,
        "total_s1_count": len(s1_ids),
        "zero_gt_count": zero_gt_count,
        "one_gt_count": one_gt_count,
        "multi_gt_count": multi_gt_count,
    }
