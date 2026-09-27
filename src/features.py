"""Pairwise feature engineering module using RapidFuzz and token analytics."""

from typing import List, Set
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token Jaccard similarity between two token lists."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / float(union)

def count_token_overlap(tokens1: List[str], tokens2: List[str]) -> int:
    """Count common tokens between two token lists."""
    return len(set(tokens1) & set(tokens2))

def count_numeric_overlap(tokens1: List[str], tokens2: List[str]) -> int:
    """Count matching numeric digit tokens between two token lists (e.g., house numbers)."""
    nums1 = {t for t in tokens1 if t.isdigit()}
    nums2 = {t for t in tokens2 if t.isdigit()}
    return len(nums1 & nums2)

def compute_numeric_agreement(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute numeric agreement indicator:
      1.0: Both have numeric tokens and they match exactly.
      0.0: Both have numeric tokens but they conflict.
      0.5: At least one side has no numeric tokens.
    """
    nums1 = {t for t in tokens1 if t.isdigit()}
    nums2 = {t for t in tokens2 if t.isdigit()}
    if not nums1 or not nums2:
        return 0.5
    if nums1 == nums2:
        return 1.0
    return 0.0

def compute_suffix_agreement(suffixes1: List[str], suffixes2: List[str]) -> float:
    """Compute legal suffix agreement indicator:
      1.0: Both have legal suffixes and they share at least one suffix.
      0.8: Neither side has a legal suffix.
      0.5: One side has a suffix, the other does not.
      0.0: Both have legal suffixes but they conflict.
    """
    set1, set2 = set(suffixes1), set(suffixes2)
    if not set1 and not set2:
        return 0.8
    if not set1 or not set2:
        return 0.5
    if set1 & set2:
        return 1.0
    return 0.0

def extract_pairwise_features(
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame
) -> pd.DataFrame:
    """Extract deterministic similarity features for each candidate pair.
    
    Args:
        candidate_pairs_df: Candidate pairs from blocking module.
        s1_df: Source 1 normalized DataFrame.
        s2_df: Source 2 normalized DataFrame.
        s3_df: Source 3 normalized DataFrame.
        
    Returns:
        DataFrame containing pair IDs, candidate source, and extracted numerical features.
    """
    if candidate_pairs_df.empty:
        return pd.DataFrame()

    # Index DataFrames by entity_id for fast O(1) lookup
    s1_map = s1_df.set_index("entity_id").to_dict("index")
    s2_map = s2_df.set_index("entity_id").to_dict("index")
    s3_map = s3_df.set_index("entity_id").to_dict("index")
    
    feature_rows = []
    
    for _, row in candidate_pairs_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        cand_src = row["candidate_source"]
        rules_str = str(row["blocking_rules"]) if pd.notna(row.get("blocking_rules")) else ""
        
        s1_data = s1_map.get(s1_id, {})
        cand_data = s2_map.get(cand_id, {}) if cand_src == "source2" else s3_map.get(cand_id, {})
        
        # Strings
        s1_name = s1_data.get("name_normalized", "")
        cand_name = cand_data.get("name_normalized", "")
        
        s1_addr = s1_data.get("address_normalized", "")
        cand_addr = cand_data.get("address_normalized", "")
        
        s1_country = s1_data.get("country_normalized", "")
        cand_country = cand_data.get("country_normalized", "")
        
        # Tokens
        s1_ntokens = s1_data.get("name_tokens", [])
        cand_ntokens = cand_data.get("name_tokens", [])
        
        s1_atokens = s1_data.get("address_tokens", [])
        cand_atokens = cand_data.get("address_tokens", [])
        
        s1_core_tokens = s1_data.get("name_core_tokens", [])
        cand_core_tokens = cand_data.get("name_core_tokens", [])
        
        s1_suffix_tokens = s1_data.get("name_suffix_tokens", [])
        cand_suffix_tokens = cand_data.get("name_suffix_tokens", [])
        
        # Name Features
        name_exact = 1.0 if s1_name == cand_name and s1_name != "" else 0.0
        name_jaccard = jaccard_similarity(s1_ntokens, cand_ntokens)
        name_char_sim = fuzz.ratio(s1_name, cand_name) / 100.0 if (s1_name or cand_name) else 0.0
        name_token_sort_sim = (
            fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
            if (s1_name or cand_name) else 0.0
        )
        name_edit_sim = distance.Levenshtein.normalized_similarity(s1_name, cand_name) if (s1_name or cand_name) else 0.0
        name_token_overlap = count_token_overlap(s1_ntokens, cand_ntokens)
        name_len_diff = abs(len(s1_name) - len(cand_name))
        name_len_ratio = (min(len(s1_name), len(cand_name)) / float(max(len(s1_name), len(cand_name)))) if max(len(s1_name), len(cand_name)) > 0 else 0.0
        
        # Core Name & Suffix Features
        core_name_token_overlap = count_token_overlap(s1_core_tokens, cand_core_tokens)
        core_name_jaccard = jaccard_similarity(s1_core_tokens, cand_core_tokens)
        suffix_agreement = compute_suffix_agreement(s1_suffix_tokens, cand_suffix_tokens)
        
        # Address Features
        address_exact = 1.0 if s1_addr == cand_addr and s1_addr != "" else 0.0
        address_jaccard = jaccard_similarity(s1_atokens, cand_atokens)
        address_char_sim = fuzz.ratio(s1_addr, cand_addr) / 100.0 if (s1_addr or cand_addr) else 0.0
        address_edit_sim = distance.Levenshtein.normalized_similarity(s1_addr, cand_addr) if (s1_addr or cand_addr) else 0.0
        address_token_overlap = count_token_overlap(s1_atokens, cand_atokens)
        address_len_diff = abs(len(s1_addr) - len(cand_addr))
        address_len_ratio = (min(len(s1_addr), len(cand_addr)) / float(max(len(s1_addr), len(cand_addr)))) if max(len(s1_addr), len(cand_addr)) > 0 else 0.0
        numeric_overlap = count_numeric_overlap(s1_atokens, cand_atokens)
        numeric_agreement = compute_numeric_agreement(s1_atokens, cand_atokens)
        
        # Country Feature
        country_exact = 1.0 if s1_country == cand_country and s1_country != "" else 0.0
        
        # Blocking Rules Feature
        rules_list = [r.strip() for r in rules_str.replace(",", "|").split("|") if r.strip()]
        blocking_rules_count = len(rules_list)
        
        feature_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "candidate_source": cand_src,
            "blocking_rules": rules_str,
            "blocking_rules_count": blocking_rules_count,
            "name_exact": name_exact,
            "name_jaccard": name_jaccard,
            "name_char_similarity": name_char_sim,
            "name_edit_similarity": name_edit_sim,
            "name_token_overlap": name_token_overlap,
            "name_len_diff": name_len_diff,
            "name_len_ratio": name_len_ratio,
            "core_name_token_overlap": core_name_token_overlap,
            "core_name_jaccard": core_name_jaccard,
            "name_token_sort_similarity": name_token_sort_sim,
            "suffix_agreement": suffix_agreement,
            "address_exact": address_exact,
            "address_jaccard": address_jaccard,
            "address_char_similarity": address_char_sim,
            "address_edit_similarity": address_edit_sim,
            "address_token_overlap": address_token_overlap,
            "address_len_diff": address_len_diff,
            "address_len_ratio": address_len_ratio,
            "numeric_token_overlap": numeric_overlap,
            "numeric_agreement_indicator": numeric_agreement,
            "country_match": country_exact,
        })
        
    return pd.DataFrame(feature_rows)
