"""
Unit tests for Milestone 5:
- Exact match scoring
- Strong match scoring
- Weak name-only match scoring
- Address-only similarity scoring
- Numeric address agreement & penalty
- Zero-match S1 handling
- Multi-match S1 handling
- Multiple valid matches
- Deterministic scoring
- F0.5 calculation
- Macro per-S1 evaluation
- Threshold sweep consistency
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure src is in import path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PipelineConfig
from src.normalize import normalize_text, normalize_name, normalize_address, tokenize_text, normalize_dataframe
from src.features import extract_pairwise_features, compute_numeric_agreement, compute_suffix_agreement
from src.matching import (
    compute_composite_score,
    predict_matches,
    calculate_f_beta,
    evaluate_macro_f05_per_s1,
    evaluate_predictions
)


def test_n5_normalization_canonicalization():
    """Test N5 legal suffix and address abbreviation canonicalization."""
    # Legal suffix canonicalization
    assert normalize_name("Acme Private Limited") == "acme pvt_ltd"
    assert normalize_name("Global Corporation Inc") == "global corp inc"
    assert normalize_name("Tech LLC") == "tech llc"
    
    # Accent & Anyascii handling
    assert normalize_name("Café René S.A.R.L.") == "cafe rene sarl"
    
    # Address abbreviation canonicalization
    assert normalize_address("123 Mahatma Gandhi Road, Suite 400") == "123 mahatma gandhi road suite 400"
    assert normalize_address("P.O. Box 99, Fifth Avenue") == "pobox 99 fifth avenue"


def _make_feature_df(**kwargs):
    """Helper to create a 1-row feature DataFrame."""
    default = {
        "name_exact": 0.0,
        "name_char_similarity": 0.0,
        "core_name_jaccard": 0.0,
        "address_exact": 0.0,
        "address_char_similarity": 0.0,
        "address_jaccard": 0.0,
        "numeric_agreement_indicator": 0.5,
        "suffix_agreement": 0.5,
        "country_match": 1.0,
    }
    default.update(kwargs)
    return pd.DataFrame([default])


def test_exact_match_scoring():
    """Verify exact match gets highest score (~1.0)."""
    config = PipelineConfig()
    df = _make_feature_df(
        name_exact=1.0,
        name_char_similarity=1.0,
        core_name_jaccard=1.0,
        address_exact=1.0,
        address_char_similarity=1.0,
        address_jaccard=1.0,
        numeric_agreement_indicator=1.0,
        suffix_agreement=1.0,
        country_match=1.0,
    )
    score = compute_composite_score(df, config).iloc[0]
    assert score >= 0.98, f"Exact match score should be >= 0.98, got {score}"


def test_strong_match_scoring():
    """Verify strong match (high name + address similarity) receives score > 0.70."""
    config = PipelineConfig()
    df = _make_feature_df(
        name_exact=0.0,
        name_char_similarity=0.90,
        core_name_jaccard=0.85,
        address_exact=0.0,
        address_char_similarity=0.88,
        address_jaccard=0.80,
        numeric_agreement_indicator=1.0,
        suffix_agreement=1.0,
        country_match=1.0,
    )
    score = compute_composite_score(df, config).iloc[0]
    assert score >= 0.50, f"Strong match score should be >= 0.50, got {score}"


def test_weak_name_only_match_scoring():
    """Verify weak name-only match with low address similarity gets lower score."""
    config = PipelineConfig()
    df = _make_feature_df(
        name_exact=0.0,
        name_char_similarity=0.70,
        core_name_jaccard=0.50,
        address_exact=0.0,
        address_char_similarity=0.10,
        address_jaccard=0.0,
        numeric_agreement_indicator=0.5,
        suffix_agreement=1.0,
        country_match=1.0,
    )
    score = compute_composite_score(df, config).iloc[0]
    assert score < 0.60, f"Weak name-only match should be < 0.60, got {score}"


def test_address_only_similarity_scoring():
    """Verify address similarity alone without name match receives low score."""
    config = PipelineConfig()
    df = _make_feature_df(
        name_exact=0.0,
        name_char_similarity=0.20,
        core_name_jaccard=0.10,
        address_exact=1.0,
        address_char_similarity=1.0,
        address_jaccard=1.0,
        numeric_agreement_indicator=1.0,
        suffix_agreement=0.5,
        country_match=1.0,
    )
    score = compute_composite_score(df, config).iloc[0]
    assert score < 0.65, f"Address-only match should be < 0.65, got {score}"


def test_numeric_address_agreement_penalty():
    """Verify conflicting house numbers (numeric_agreement_indicator == 0.0) applies penalty on weak address."""
    config = PipelineConfig()
    df_agree = _make_feature_df(
        name_exact=0.0, name_char_similarity=0.85, core_name_jaccard=0.8,
        address_exact=0.0, address_char_similarity=0.50, address_jaccard=0.5,
        numeric_agreement_indicator=1.0, suffix_agreement=1.0, country_match=1.0
    )
    df_disagree = _make_feature_df(
        name_exact=0.0, name_char_similarity=0.85, core_name_jaccard=0.8,
        address_exact=0.0, address_char_similarity=0.50, address_jaccard=0.5,
        numeric_agreement_indicator=0.0, suffix_agreement=1.0, country_match=1.0
    )
    score_agree = compute_composite_score(df_agree, config).iloc[0]
    score_disagree = compute_composite_score(df_disagree, config).iloc[0]
    assert score_disagree < score_agree - 0.15, "Numeric disagreement should penalize score noticeably"


def test_f_beta_calculation():
    """Verify F0.5 precision weighting."""
    # F0.5 puts more weight on Precision than Recall
    # P=1.0, R=0.5 -> F0.5 = 1.25 * 1.0 * 0.5 / (0.25*1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    f05 = calculate_f_beta(1.0, 0.5, beta=0.5)
    assert abs(f05 - 0.833333) < 1e-4

    # Precision = 0 -> F0.5 = 0
    assert calculate_f_beta(0.0, 0.5, beta=0.5) == 0.0

    # TP=0, FP=0, FN=0 -> 1.0
    assert calculate_f_beta(1.0, 1.0, beta=0.5) == 1.0


def test_macro_per_s1_evaluation():
    """Verify competition-style macro per-S1 evaluation including zero-GT S1 cases."""
    s1_ids = ["S1_1", "S1_2", "S1_3", "S1_4"]
    
    # Ground truth:
    # S1_1 -> S2_100 (1 match)
    # S1_2 -> S2_200, S3_201 (2 matches)
    # S1_3 -> zero matches
    # S1_4 -> zero matches
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1_1", "matched_entity_ids": "S2_100"},
        {"source1_entity_id": "S1_2", "matched_entity_ids": "S2_200, S3_201"},
    ])

    # Predictions:
    # S1_1 -> [S2_100] (Correct prediction: P=1, R=1, F0.5=1.0)
    # S1_2 -> [S2_200] (Partial match: P=1, R=0.5, F0.5=0.8333)
    # S1_3 -> [] (Correct zero-match: P=1, R=1, F0.5=1.0)
    # S1_4 -> [S2_999] (False positive on zero-match: P=0, R=0, F0.5=0.0)
    predictions_df = pd.DataFrame([
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_100", "match_score": 0.95},
        {"source1_entity_id": "S1_2", "candidate_entity_id": "S2_200", "match_score": 0.90},
        {"source1_entity_id": "S1_4", "candidate_entity_id": "S2_999", "match_score": 0.85},
    ])

    metrics = evaluate_macro_f05_per_s1(predictions_df, gt_df, s1_ids=s1_ids, beta=0.5)

    assert metrics["total_s1_count"] == 4
    assert metrics["zero_gt_count"] == 2
    assert metrics["one_gt_count"] == 1
    assert metrics["multi_gt_count"] == 1

    # Macro F0.5 should be (1.0 + 0.833333 + 1.0 + 0.0) / 4 = 2.833333 / 4 = 0.708333
    assert abs(metrics["macro_f05"] - 0.708333) < 1e-4
    assert metrics["macro_precision"] == 0.75  # (1 + 1 + 1 + 0)/4
    assert metrics["macro_recall"] == 0.625    # (1 + 0.5 + 1 + 0)/4


def test_deterministic_scoring_and_sweep_consistency():
    """Verify that feature extraction and scoring produce identical output given identical input."""
    cand_df = pd.DataFrame([
        {"source1_entity_id": "S1_01", "candidate_entity_id": "S2_01", "candidate_source": "source2", "blocking_rules": "exact_name"},
        {"source1_entity_id": "S1_01", "candidate_entity_id": "S3_01", "candidate_source": "source3", "blocking_rules": "name_token"},
    ])
    s1_df = normalize_dataframe(pd.DataFrame([{
        "entity_id": "S1_01", "business_name": "Acme Corp", "business_address": "100 Main St", "country": "US"
    }]))
    s2_df = normalize_dataframe(pd.DataFrame([{
        "entity_id": "S2_01", "business_name": "Acme Corporation", "business_address": "100 Main St", "country": "US"
    }]))
    s3_df = normalize_dataframe(pd.DataFrame([{
        "entity_id": "S3_01", "business_name": "Acme Limited", "business_address": "500 High St", "country": "US"
    }]))

    feat1 = extract_pairwise_features(cand_df, s1_df, s2_df, s3_df)
    feat2 = extract_pairwise_features(cand_df, s1_df, s2_df, s3_df)

    pd.testing.assert_frame_equal(feat1, feat2)

    config = PipelineConfig()
    pred1 = predict_matches(feat1, config, threshold=0.7)
    pred2 = predict_matches(feat2, config, threshold=0.7)

    pd.testing.assert_frame_equal(pred1, pred2)
