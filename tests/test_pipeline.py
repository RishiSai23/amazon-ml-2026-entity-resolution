"""Unit tests verifying pipeline components, formatting rules, and invariants."""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PipelineConfig
from src.normalize import normalize_text, tokenize_text, normalize_dataframe
from src.blocking import generate_candidate_pairs
from src.features import extract_pairwise_features, jaccard_similarity, count_numeric_overlap
from src.matching import predict_matches, evaluate_predictions
from src.postprocess import format_matching_results, validate_pipeline_outputs

def test_normalization():
    """Verify non-destructive text normalization."""
    raw = "ABC Technologies Pvt. Ltd. (Branch #12)"
    norm = normalize_text(raw)
    assert norm == "abc technologies pvt ltd branch 12"
    
    tokens = tokenize_text(norm)
    assert tokens == ["abc", "technologies", "pvt", "ltd", "branch", "12"]

def test_dataframe_normalization():
    """Verify DataFrame normalization produces required columns."""
    df = pd.DataFrame([{
        "entity_id": "S1_001",
        "business_name": "ABC Tech Ltd.",
        "business_address": "12 Main St",
        "country": "India"
    }])
    norm_df = normalize_dataframe(df)
    assert "name_normalized" in norm_df.columns
    assert "address_normalized" in norm_df.columns
    assert "name_tokens" in norm_df.columns
    assert "name_suffix_tokens" in norm_df.columns
    assert norm_df["name_suffix_tokens"].iloc[0] == ["ltd"]

def test_feature_extraction():
    """Test Jaccard and numeric digit token overlap features."""
    t1 = ["12", "mg", "road"]
    t2 = ["12", "mahatma", "gandhi", "rd"]
    
    jacc = jaccard_similarity(t1, t2)
    assert 0.0 < jacc < 1.0
    
    num_overlap = count_numeric_overlap(t1, t2)
    assert num_overlap == 1  # Matches "12"

def test_singleton_and_multimatch_formatting():
    """Verify formatting handles singletons (empty matches) and multi-matches cleanly."""
    s1_df = pd.DataFrame([
        {"entity_id": "S1_01"},
        {"entity_id": "S1_02"},
    ])
    predictions_df = pd.DataFrame([
        {"source1_entity_id": "S1_01", "candidate_entity_id": "S2_10", "candidate_source": "source2", "match_score": 0.95},
        {"source1_entity_id": "S1_01", "candidate_entity_id": "S3_20", "candidate_source": "source3", "match_score": 0.90},
    ])
    
    results_df = format_matching_results(s1_df, predictions_df)
    
    # S1_01 multi-match sorted
    row_01 = results_df[results_df["entity_id"] == "S1_01"].iloc[0]
    assert row_01["matched_entity_ids"] == "S2_10,S3_20"
    
    # S1_02 singleton (no match -> empty string)
    row_02 = results_df[results_df["entity_id"] == "S1_02"].iloc[0]
    assert row_02["matched_entity_ids"] == ""

def test_candidate_pair_invariant():
    """Verify assertion fails if predicted match is not in candidate set."""
    s1_df = pd.DataFrame([{"entity_id": "S1_01"}])
    s2_df = pd.DataFrame([{"entity_id": "S2_10"}])
    s3_df = pd.DataFrame([])
    
    candidate_pairs_df = pd.DataFrame([
        {"source1_entity_id": "S1_01", "candidate_entity_id": "S2_99", "candidate_source": "source2", "blocking_rules": "exact"}
    ])
    results_df = pd.DataFrame([
        {"entity_id": "S1_01", "matched_entity_ids": "S2_10"}
    ])
    
    with pytest.raises(AssertionError, match="NOT present in candidate_pairs.tsv"):
        validate_pipeline_outputs(s1_df, s2_df, s3_df, candidate_pairs_df, results_df)

def test_end_to_end_mock_pipeline():
    """Run end-to-end integration test on mock dataset."""
    config = PipelineConfig()
    
    # Load mock train
    from src.io_utils import load_source_dataset
    s1_raw, s2_raw, s3_raw, gt_df = load_source_dataset(config.train_dir, is_train=True, required_cols=config.required_columns)
    
    s1_norm = normalize_dataframe(s1_raw, config.legal_suffixes)
    s2_norm = normalize_dataframe(s2_raw, config.legal_suffixes)
    s3_norm = normalize_dataframe(s3_raw, config.legal_suffixes)
    
    candidate_pairs_df = generate_candidate_pairs(s1_norm, s2_norm, s3_norm, config)
    assert not candidate_pairs_df.empty
    
    features_df = extract_pairwise_features(candidate_pairs_df, s1_norm, s2_norm, s3_norm)
    assert len(features_df) == len(candidate_pairs_df)
    
    predictions_df = predict_matches(features_df, config)
    
    metrics = evaluate_predictions(candidate_pairs_df, predictions_df, gt_df, beta=0.5)
    assert metrics["candidate_recall"] == 1.0
    assert metrics["f0.5"] > 0.8
    
    results_df = format_matching_results(s1_norm, predictions_df)
    assert len(results_df) == len(s1_raw)
    
    val_ok = validate_pipeline_outputs(s1_norm, s2_norm, s3_norm, candidate_pairs_df, results_df)
    assert val_ok is True
