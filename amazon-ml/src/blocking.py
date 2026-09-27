"""Multi-strategy candidate generation (blocking) module."""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from src.config import PipelineConfig

def generate_char_signature(name_norm: str, sig_len: int = 3) -> str:
    """Generate a lightweight character signature for blocking."""
    clean_chars = [c for c in name_norm if c.isalnum()]
    if not clean_chars:
        return ""
    return "".join(sorted(clean_chars[:sig_len]))

def build_blocking_indices(
    candidates_df: pd.DataFrame, 
    candidate_source_label: str,
    config: PipelineConfig
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Build inverted index lookup tables for a candidate DataFrame using fast tuple iteration."""
    exact_name_idx = defaultdict(set)
    name_token_idx = defaultdict(set)
    address_token_idx = defaultdict(set)
    char_sig_idx = defaultdict(set)
    
    if candidates_df.empty:
        return exact_name_idx, name_token_idx, address_token_idx, char_sig_idx
        
    ids = candidates_df["entity_id"].values
    countries = candidates_df["country_normalized"].values
    names = candidates_df["name_normalized"].values
    core_tokens_col = candidates_df["name_core_tokens"].values
    name_tokens_col = candidates_df["name_tokens"].values
    addr_tokens_col = candidates_df["address_tokens"].values
    
    min_token_len = config.min_token_len
    min_addr_len = min_token_len + 1
    char_sig_len = config.char_sig_len
    enable_exact = config.enable_exact_name_block
    enable_name_prefix = config.enable_name_prefix_block
    enable_addr_token = config.enable_address_token_block
    enable_csig = config.enable_char_signature_block
    
    for cand_id, country, name, core_tokens, name_tokens, addr_tokens in zip(
        ids, countries, names, core_tokens_col, name_tokens_col, addr_tokens_col
    ):
        cand_tuple = (cand_id, candidate_source_label)
        
        # Block A: Exact Name
        if enable_exact and name:
            exact_name_idx[(country, name)].add(cand_tuple)
            
        # Block B: Name Token
        if enable_name_prefix:
            tokens = core_tokens if (core_tokens is not None and len(core_tokens) > 0) else name_tokens
            if tokens:
                for token in tokens:
                    if len(token) >= min_token_len:
                        name_token_idx[(country, token)].add(cand_tuple)
                        
        # Block C: Address Token
        if enable_addr_token and addr_tokens:
            for token in addr_tokens:
                if token.isdigit() or len(token) >= min_addr_len:
                    address_token_idx[(country, token)].add(cand_tuple)
                    
        # Block D: Character Signature
        if enable_csig and name:
            clean_chars = [c for c in name if c.isalnum()]
            if clean_chars:
                sig = "".join(sorted(clean_chars[:char_sig_len]))
                char_sig_idx[(country, sig)].add(cand_tuple)
                
    return exact_name_idx, name_token_idx, address_token_idx, char_sig_idx

def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config: PipelineConfig
) -> pd.DataFrame:
    """Generate candidate pairs combining Source 1 against Source 2 and Source 3 using multi-strategy blocking."""
    # Build candidate lookup indices for Source 2 and Source 3
    s2_exact, s2_ntoken, s2_atoken, s2_csig = build_blocking_indices(s2_df, "source2", config)
    s3_exact, s3_ntoken, s3_atoken, s3_csig = build_blocking_indices(s3_df, "source3", config)
    
    candidate_map = defaultdict(set)
    
    s1_ids = s1_df["entity_id"].values
    s1_countries = s1_df["country_normalized"].values
    s1_names = s1_df["name_normalized"].values
    s1_core_tokens_col = s1_df["name_core_tokens"].values
    s1_name_tokens_col = s1_df["name_tokens"].values
    s1_addr_tokens_col = s1_df["address_tokens"].values
    
    min_token_len = config.min_token_len
    min_addr_len = min_token_len + 1
    char_sig_len = config.char_sig_len
    
    MAX_BLOCK_KEY_SIZE = 500

    for s1_id, country, name, core_tokens, name_tokens, addr_tokens in zip(
        s1_ids, s1_countries, s1_names, s1_core_tokens_col, s1_name_tokens_col, s1_addr_tokens_col
    ):
        # Strategy 1: Exact Name Match
        if config.enable_exact_name_block and name:
            key = (country, name)
            cands = s2_exact.get(key, set()) | s3_exact.get(key, set())
            for cand_tuple in cands:
                candidate_map[(s1_id, cand_tuple[0], cand_tuple[1])].add("exact_name")
                
        # Strategy 2: Name Token Match
        if config.enable_name_prefix_block:
            tokens = core_tokens if (core_tokens is not None and len(core_tokens) > 0) else name_tokens
            if tokens:
                for token in tokens:
                    if len(token) >= min_token_len:
                        key = (country, token)
                        s2_matches = s2_ntoken.get(key, set())
                        s3_matches = s3_ntoken.get(key, set())
                        if len(s2_matches) + len(s3_matches) <= MAX_BLOCK_KEY_SIZE:
                            for cand_tuple in s2_matches | s3_matches:
                                candidate_map[(s1_id, cand_tuple[0], cand_tuple[1])].add("name_token")
                            
        # Strategy 3: Address Token Match
        if config.enable_address_token_block and addr_tokens:
            for token in addr_tokens:
                if token.isdigit() or len(token) >= min_addr_len:
                    key = (country, token)
                    s2_matches = s2_atoken.get(key, set())
                    s3_matches = s3_atoken.get(key, set())
                    if len(s2_matches) + len(s3_matches) <= MAX_BLOCK_KEY_SIZE:
                        for cand_tuple in s2_matches | s3_matches:
                            candidate_map[(s1_id, cand_tuple[0], cand_tuple[1])].add("address_token")
                        
        # Strategy 4: Character Signature Match
        if config.enable_char_signature_block and name:
            clean_chars = [c for c in name if c.isalnum()]
            if clean_chars:
                sig = "".join(sorted(clean_chars[:char_sig_len]))
                key = (country, sig)
                s2_matches = s2_csig.get(key, set())
                s3_matches = s3_csig.get(key, set())
                if len(s2_matches) + len(s3_matches) <= MAX_BLOCK_KEY_SIZE:
                    for cand_tuple in s2_matches | s3_matches:
                        candidate_map[(s1_id, cand_tuple[0], cand_tuple[1])].add("char_signature")
                    
    candidate_rows = []
    for (s1_id, cand_id, cand_src), rules_set in candidate_map.items():
        sorted_rules = "|".join(sorted(rules_set))
        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "candidate_source": cand_src,
            "blocking_rules": sorted_rules
        })
        
    if not candidate_rows:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "candidate_source", "blocking_rules"])
        
    res_df = pd.DataFrame(candidate_rows)
    return res_df.sort_values(by=["source1_entity_id", "candidate_entity_id", "candidate_source"]).reset_index(drop=True)
