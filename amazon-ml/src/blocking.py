"""Multi-strategy candidate generation (blocking) module."""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from src.config import PipelineConfig

def generate_char_signature(name_norm: str, sig_len: int = 3) -> str:
    """Generate a lightweight character signature for blocking.
    Takes first sig_len unique characters or sorted characters from normalized name.
    
    Args:
        name_norm: Normalized business name.
        sig_len: Signature length.
        
    Returns:
        Character signature string.
    """
    clean_chars = [c for c in name_norm if c.isalnum()]
    if not clean_chars:
        return ""
    # Use first sig_len characters sorted
    return "".join(sorted(clean_chars[:sig_len]))

def build_blocking_indices(
    candidates_df: pd.DataFrame, 
    candidate_source_label: str,
    config: PipelineConfig
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Build inverted index lookup tables for a candidate DataFrame (Source 2 or Source 3).
    
    Args:
        candidates_df: Normalized candidate DataFrame.
        candidate_source_label: Identifier string ("source2" or "source3").
        config: PipelineConfig object.
        
    Returns:
        Tuple of (exact_name_idx, name_token_idx, address_token_idx, char_sig_idx)
    """
    exact_name_idx = defaultdict(set)
    name_token_idx = defaultdict(set)
    address_token_idx = defaultdict(set)
    char_sig_idx = defaultdict(set)
    
    for _, row in candidates_df.iterrows():
        cand_id = row["entity_id"]
        country = row["country_normalized"]
        cand_tuple = (cand_id, candidate_source_label)
        
        # Block A: Exact Name
        if config.enable_exact_name_block and row["name_normalized"]:
            key = (country, row["name_normalized"])
            exact_name_idx[key].add(cand_tuple)
            
        # Block B: Name Token
        if config.enable_name_prefix_block:
            tokens = row["name_core_tokens"] if row["name_core_tokens"] else row["name_tokens"]
            for token in tokens:
                if len(token) >= config.min_token_len:
                    key = (country, token)
                    name_token_idx[key].add(cand_tuple)
                    
        # Block C: Address Token
        if config.enable_address_token_block:
            for token in row["address_tokens"]:
                # Prioritize numeric house/street numbers or long address tokens
                if token.isdigit() or len(token) >= config.min_token_len + 1:
                    key = (country, token)
                    address_token_idx[key].add(cand_tuple)
                    
        # Block D: Character Signature
        if config.enable_char_signature_block and row["name_normalized"]:
            sig = generate_char_signature(row["name_normalized"], config.char_sig_len)
            if sig:
                key = (country, sig)
                char_sig_idx[key].add(cand_tuple)
                
    return exact_name_idx, name_token_idx, address_token_idx, char_sig_idx

def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config: PipelineConfig
) -> pd.DataFrame:
    """Generate candidate pairs combining Source 1 against Source 2 and Source 3 using multi-strategy blocking.
    
    Args:
        s1_df: Normalized Source 1 DataFrame.
        s2_df: Normalized Source 2 DataFrame.
        s3_df: Normalized Source 3 DataFrame.
        config: PipelineConfig object.
        
    Returns:
        DataFrame containing candidate pairs with columns:
        ['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocking_rules']
    """
    # Build candidate lookup indices for Source 2 and Source 3
    s2_exact, s2_ntoken, s2_atoken, s2_csig = build_blocking_indices(s2_df, "source2", config)
    s3_exact, s3_ntoken, s3_atoken, s3_csig = build_blocking_indices(s3_df, "source3", config)
    
    candidate_rows = []
    
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        country = s1_row["country_normalized"]
        
        # Map: (candidate_id, candidate_source) -> set of rule labels
        matched_candidates: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        
        # 1. Exact Name Matching
        if config.enable_exact_name_block and s1_row["name_normalized"]:
            key = (country, s1_row["name_normalized"])
            for cand in s2_exact.get(key, []):
                matched_candidates[cand].add("exact_name")
            for cand in s3_exact.get(key, []):
                matched_candidates[cand].add("exact_name")
                
        # 2. Name Token Matching
        if config.enable_name_prefix_block:
            tokens = s1_row["name_core_tokens"] if s1_row["name_core_tokens"] else s1_row["name_tokens"]
            for token in tokens:
                if len(token) >= config.min_token_len:
                    key = (country, token)
                    for cand in s2_ntoken.get(key, []):
                        matched_candidates[cand].add("name_token")
                    for cand in s3_ntoken.get(key, []):
                        matched_candidates[cand].add("name_token")
                        
        # 3. Address Token Matching
        if config.enable_address_token_block:
            for token in s1_row["address_tokens"]:
                if token.isdigit() or len(token) >= config.min_token_len + 1:
                    key = (country, token)
                    for cand in s2_atoken.get(key, []):
                        matched_candidates[cand].add("address_token")
                    for cand in s3_atoken.get(key, []):
                        matched_candidates[cand].add("address_token")
                        
        # 4. Character Signature Matching
        if config.enable_char_signature_block and s1_row["name_normalized"]:
            sig = generate_char_signature(s1_row["name_normalized"], config.char_sig_len)
            if sig:
                key = (country, sig)
                for cand in s2_csig.get(key, []):
                    matched_candidates[cand].add("char_signature")
                for cand in s3_csig.get(key, []):
                    matched_candidates[cand].add("char_signature")
                    
        # Flatten candidates into rows
        for (cand_id, cand_src), rules in matched_candidates.items():
            candidate_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": cand_src,
                "blocking_rules": "|".join(sorted(rules))
            })
            
    candidate_pairs_df = pd.DataFrame(candidate_rows)
    if candidate_pairs_df.empty:
        candidate_pairs_df = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id", "candidate_source", "blocking_rules"
        ])
        
    return candidate_pairs_df
