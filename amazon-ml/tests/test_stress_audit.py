"""Comprehensive Milestone 2 Audit & Stress Test Suite.

Tests cover:
  1. Normalization edge cases (punctuation, accents, short names, numbers, empty inputs).
  2. Blocking recovery matrix across noisy business name & address variations.
  3. Controlled candidate pair matcher scoring table.
  4. Exact F0.5 formula & edge case evaluation.
  5. Determinism and output contract validation.
  6. Runtime performance breakdown.
"""

import sys
import time
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PipelineConfig
from src.normalize import normalize_text, tokenize_text, normalize_dataframe
from src.blocking import generate_candidate_pairs, generate_char_signature
from src.features import extract_pairwise_features
from src.matching import predict_matches, evaluate_predictions, compute_composite_score
from src.postprocess import format_matching_results, validate_pipeline_outputs
from src.io_utils import load_source_dataset

# =====================================================================
# 1. Normalization Stress Tests
# =====================================================================

@pytest.mark.parametrize("input_text,expected", [
    ("ABC Pvt. Ltd.", "abc pvt ltd"),
    ("ABC Private Limited", "abc private limited"),
    ("ABC TECHNOLOGIES", "abc technologies"),
    ("abc technologies", "abc technologies"),
    ("12 MG Road", "12 mg road"),
    ("12 Mahatma Gandhi Road", "12 mahatma gandhi road"),
    ("St. Mary's School", "st mary s school"),
    ("St Marys School", "st marys school"),
    ("ABC-TECH", "abc tech"),
    ("José Café", "jose cafe"),
    ("   Multiple   Spaces   ", "multiple spaces"),
    ("Special!!! Characters@#$%^&*", "special characters"),
    ("12345 67890", "12345 67890"),
    ("   ", ""),
    (None, ""),
    (np.nan, ""),
    ("LTD PVT INC", "ltd pvt inc"),
    ("HP", "hp"),
])
def test_normalization_robustness(input_text, expected):
    """Verify normalization returns expected string for edge cases."""
    result = normalize_text(input_text)
    assert result == expected

# =====================================================================
# 2. Blocking Strategy Recovery Matrix Test
# =====================================================================

def test_blocking_strategy_matrix():
    """Verify which blocking strategies recover true pairs under various noise types."""
    config = PipelineConfig()
    
    # Source 1 baseline entities
    s1_data = [
        {"entity_id": "S1_01", "business_name": "ABC Technologies Pvt Ltd", "business_address": "12 MG Road Bangalore", "country": "India"},
        {"entity_id": "S1_02", "business_name": "Global Logistics Express Inc", "business_address": "456 Industrial Park Sector 5", "country": "USA"},
        {"entity_id": "S1_03", "business_name": "St Mary School", "business_address": "100 Baker Street", "country": "UK"},
    ]
    
    # Source 2 noisy variations
    s2_data = [
        # S1_01 variations
        {"entity_id": "S2_101", "business_name": "ABC Technologies Pvt Ltd", "business_address": "12 MG Road Bangalore", "country": "India"}, # Exact
        {"entity_id": "S2_102", "business_name": "ABX Technologies Pvt Ltd", "business_address": "12 MG Road Bangalore", "country": "India"}, # Typo in name
        {"entity_id": "S2_103", "business_name": "Technologies ABC Pvt Ltd", "business_address": "12 MG Road Bangalore", "country": "India"}, # Word order change
        {"entity_id": "S2_104", "business_name": "ABC Tech Private Limited", "business_address": "12 Mahatma Gandhi Rd", "country": "India"}, # Abbreviation & address noise
        {"entity_id": "S2_105", "business_name": "ABC Technologies", "business_address": "Bangalore", "country": "India"},                     # Missing address component
        {"entity_id": "S2_106", "business_name": "AyBeeCee Technologies", "business_address": "12 MG Rd Bangalore", "country": "India"},       # Transliteration
    ]
    
    s1_df = normalize_dataframe(pd.DataFrame(s1_data), config.legal_suffixes)
    s2_df = normalize_dataframe(pd.DataFrame(s2_data), config.legal_suffixes)
    s3_df = normalize_dataframe(pd.DataFrame([]), config.legal_suffixes)
    
    cand_pairs = generate_candidate_pairs(s1_df, s2_df, s3_df, config)
    
    # Build lookup: candidate_entity_id -> set of rules
    recovered_map = {}
    for _, row in cand_pairs[cand_pairs["source1_entity_id"] == "S1_01"].iterrows():
        recovered_map[row["candidate_entity_id"]] = set(row["blocking_rules"].split("|"))
        
    print("\n--- Blocking Recovery Matrix Report ---")
    print(f"Exact Match (S2_101):         {recovered_map.get('S2_101', 'MISSED')}")
    print(f"Misspelled Name (S2_102):     {recovered_map.get('S2_102', 'MISSED')}")
    print(f"Word Order Change (S2_103):   {recovered_map.get('S2_103', 'MISSED')}")
    print(f"Abbreviation & Addr (S2_104): {recovered_map.get('S2_104', 'MISSED')}")
    print(f"Missing Address (S2_105):     {recovered_map.get('S2_105', 'MISSED')}")
    print(f"Transliteration (S2_106):     {recovered_map.get('S2_106', 'MISSED')}")
    
    # Assert true matches recovered by at least one strategy
    assert "S2_101" in recovered_map
    assert "S2_102" in recovered_map
    assert "S2_103" in recovered_map

# =====================================================================
# 3. Matcher Controlled Candidate Pairs Scoring Test
# =====================================================================

def test_matcher_controlled_scoring_table():
    """Test matcher scoring behavior across 5 controlled scenarios."""
    config = PipelineConfig(match_threshold=0.70)
    
    test_cases = [
        {
            "case": "1. Very Strong Match",
            "s1_name": "abc technologies pvt ltd", "cand_name": "abc technology private limited",
            "s1_addr": "12 mg road bangalore", "cand_addr": "12 mg road bangalore",
            "s1_country": "india", "cand_country": "india"
        },
        {
            "case": "2. Strong Name / Weak Address",
            "s1_name": "abc technologies pvt ltd", "cand_name": "abc technology private limited",
            "s1_addr": "12 mg road bangalore", "cand_addr": "99 industrial park chennai",
            "s1_country": "india", "cand_country": "india"
        },
        {
            "case": "3. Weak Name / Strong Address",
            "s1_name": "abc tech", "cand_name": "xyz enterprise corp",
            "s1_addr": "12 mg road bangalore", "cand_addr": "12 mg road bangalore",
            "s1_country": "india", "cand_country": "india"
        },
        {
            "case": "4. Same Name / Different Country",
            "s1_name": "abc technologies pvt ltd", "cand_name": "abc technologies pvt ltd",
            "s1_addr": "12 mg road bangalore", "cand_addr": "12 mg road london",
            "s1_country": "india", "cand_country": "uk"
        },
        {
            "case": "5. Completely Unrelated",
            "s1_name": "abc technologies pvt ltd", "cand_name": "random bakery shop",
            "s1_addr": "12 mg road bangalore", "cand_addr": "55 high street london",
            "s1_country": "india", "cand_country": "uk"
        }
    ]
    
    # Construct feature rows
    feature_rows = []
    for tc in test_cases:
        from src.features import jaccard_similarity
        from rapidfuzz import distance, fuzz
        
        name_exact = 1.0 if tc["s1_name"] == tc["cand_name"] else 0.0
        name_jaccard = jaccard_similarity(tc["s1_name"].split(), tc["cand_name"].split())
        name_char_sim = fuzz.ratio(tc["s1_name"], tc["cand_name"]) / 100.0
        
        addr_exact = 1.0 if tc["s1_addr"] == tc["cand_addr"] else 0.0
        addr_jaccard = jaccard_similarity(tc["s1_addr"].split(), tc["cand_addr"].split())
        addr_char_sim = fuzz.ratio(tc["s1_addr"], tc["cand_addr"]) / 100.0
        
        country_match = 1.0 if tc["s1_country"] == tc["cand_country"] else 0.0
        
        feature_rows.append({
            "source1_entity_id": "S1_TEST",
            "candidate_entity_id": tc["case"],
            "candidate_source": "source2",
            "blocking_rules": "test",
            "name_exact": name_exact,
            "name_jaccard": name_jaccard,
            "name_char_similarity": name_char_sim,
            "address_exact": addr_exact,
            "address_jaccard": addr_jaccard,
            "address_char_similarity": addr_char_sim,
            "country_match": country_match,
        })
        
    features_df = pd.DataFrame(feature_rows)
    scores = compute_composite_score(features_df, config)
    features_df["final_score"] = scores
    features_df["predicted_match"] = features_df["final_score"] >= config.match_threshold
    
    print("\n--- Matcher Controlled Scoring Table ---")
    for _, row in features_df.iterrows():
        print(f"{row['candidate_entity_id']:<35} | NameSim: {row['name_char_similarity']:.2f} | AddrSim: {row['address_char_similarity']:.2f} | Country: {row['country_match']:.0f} | FinalScore: {row['final_score']:.4f} | Match: {row['predicted_match']}")
        
    # Verify expected outcomes
    assert features_df.iloc[0]["predicted_match"] == True   # Very Strong Match
    assert features_df.iloc[4]["predicted_match"] == False  # Completely Unrelated

# =====================================================================
# 4. F0.5 Formula Exact Verification
# =====================================================================

def test_f05_formula_exactness():
    """Verify F0.5 calculation matches theoretical formula exactly."""
    # Test case: TP = 4, FP = 1, FN = 1
    # P = 4 / 5 = 0.8
    # R = 4 / 5 = 0.8
    # F0.5 = (1.25 * 0.8 * 0.8) / (0.25 * 0.8 + 0.8) = 0.8 / 1.0 = 0.8
    
    gt_df = pd.DataFrame([
        {"entity_id": "S1_1", "matched_entity_ids": "S2_1"},
        {"entity_id": "S1_2", "matched_entity_ids": "S2_2"},
        {"entity_id": "S1_3", "matched_entity_ids": "S2_3"},
        {"entity_id": "S1_4", "matched_entity_ids": "S2_4"},
        {"entity_id": "S1_5", "matched_entity_ids": "S2_5"}, # FN
    ])
    
    cand_pairs_df = pd.DataFrame([
        {"source1_entity_id": f"S1_{i}", "candidate_entity_id": f"S2_{i}"} for i in range(1, 6)
    ] + [{"source1_entity_id": "S1_1", "candidate_entity_id": "S2_99"}]) # Extra candidate
    
    pred_df = pd.DataFrame([
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_1"}, # TP
        {"source1_entity_id": "S1_2", "candidate_entity_id": "S2_2"}, # TP
        {"source1_entity_id": "S1_3", "candidate_entity_id": "S2_3"}, # TP
        {"source1_entity_id": "S1_4", "candidate_entity_id": "S2_4"}, # TP
        {"source1_entity_id": "S1_1", "candidate_entity_id": "S2_99"}, # FP
    ])
    
    metrics = evaluate_predictions(cand_pairs_df, pred_df, gt_df, beta=0.5)
    
    assert metrics["true_positives"] == 4
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert abs(metrics["precision"] - 0.8) < 1e-6
    assert abs(metrics["recall"] - 0.8) < 1e-6
    assert abs(metrics["f0.5"] - 0.8) < 1e-6

# =====================================================================
# 5. Determinism & Output Verification Test
# =====================================================================

def test_pipeline_determinism(tmp_path):
    """Run pipeline twice and verify candidate_pairs.tsv and matching_results.tsv are 100% identical."""
    config = PipelineConfig()
    s1_raw, s2_raw, s3_raw, gt_df = load_source_dataset(config.train_dir, is_train=True, required_cols=config.required_columns)
    
    # Run 1
    s1_norm1 = normalize_dataframe(s1_raw, config.legal_suffixes)
    s2_norm1 = normalize_dataframe(s2_raw, config.legal_suffixes)
    s3_norm1 = normalize_dataframe(s3_raw, config.legal_suffixes)
    cand_pairs1 = generate_candidate_pairs(s1_norm1, s2_norm1, s3_norm1, config)
    feat1 = extract_pairwise_features(cand_pairs1, s1_norm1, s2_norm1, s3_norm1)
    pred1 = predict_matches(feat1, config)
    res1 = format_matching_results(s1_norm1, pred1)
    
    # Run 2
    s1_norm2 = normalize_dataframe(s1_raw, config.legal_suffixes)
    s2_norm2 = normalize_dataframe(s2_raw, config.legal_suffixes)
    s3_norm2 = normalize_dataframe(s3_raw, config.legal_suffixes)
    cand_pairs2 = generate_candidate_pairs(s1_norm2, s2_norm2, s3_norm2, config)
    feat2 = extract_pairwise_features(cand_pairs2, s1_norm2, s2_norm2, s3_norm2)
    pred2 = predict_matches(feat2, config)
    res2 = format_matching_results(s1_norm2, pred2)
    
    pd.testing.assert_frame_equal(cand_pairs1, cand_pairs2)
    pd.testing.assert_frame_equal(res1, res2)

# =====================================================================
# 6. Runtime Performance Profiling
# =====================================================================

def test_runtime_profiling():
    """Profile runtime breakdown across pipeline stages."""
    config = PipelineConfig()
    s1_raw, s2_raw, s3_raw, _ = load_source_dataset(config.train_dir, is_train=True, required_cols=config.required_columns)
    
    t0 = time.perf_counter()
    s1_norm = normalize_dataframe(s1_raw, config.legal_suffixes)
    s2_norm = normalize_dataframe(s2_raw, config.legal_suffixes)
    s3_norm = normalize_dataframe(s3_raw, config.legal_suffixes)
    t_norm = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    cand_pairs = generate_candidate_pairs(s1_norm, s2_norm, s3_norm, config)
    t_block = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    features = extract_pairwise_features(cand_pairs, s1_norm, s2_norm, s3_norm)
    t_feat = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    preds = predict_matches(features, config)
    t_match = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    results = format_matching_results(s1_norm, preds)
    t_post = time.perf_counter() - t0
    
    print("\n--- Runtime Breakdown Baseline ---")
    print(f"  Normalization:      {t_norm*1000:.3f} ms")
    print(f"  Blocking:           {t_block*1000:.3f} ms")
    print(f"  Feature Extraction: {t_feat*1000:.3f} ms")
    print(f"  Matching Engine:    {t_match*1000:.3f} ms")
    print(f"  Post-processing:    {t_post*1000:.3f} ms")
    print(f"  Total Pipeline:     {(t_norm+t_block+t_feat+t_match+t_post)*1000:.3f} ms")
