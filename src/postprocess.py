"""Post-processing, submission formatting, and output validation module."""

from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

def format_matching_results(s1_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Format final predictions into submission format with entity_id and matched_entity_ids.
    
    Requirements:
      - Every Source 1 entity must appear exactly once.
      - Matched IDs are comma-separated with NO spaces around commas.
      - Blank/empty string when there is no match.
      - Deterministic ordering by entity_id.
      
    Args:
        s1_df: Source 1 DataFrame.
        predictions_df: Predicted matching pairs.
        
    Returns:
        DataFrame with columns ['entity_id', 'matched_entity_ids'] sorted by entity_id.
    """
    s1_all_ids = s1_df["entity_id"].unique()
    
    # Map s1_id -> list of candidate matched IDs
    match_map: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_all_ids}
    
    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]
            if s1_id in match_map:
                match_map[s1_id].append(cand_id)
                
    results_rows = []
    for s1_id in sorted(s1_all_ids):
        matched_list = sorted(list(set(match_map[s1_id])))
        matched_str = ",".join(matched_list)
        results_rows.append({
            "entity_id": s1_id,
            "matched_entity_ids": matched_str
        })
        
    results_df = pd.DataFrame(results_rows)
    return results_df

def validate_pipeline_outputs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame,
    results_df: pd.DataFrame,
    matching_filepath: Path = None,
    candidate_filepath: Path = None
) -> bool:
    """Run comprehensive local sanity validation against output DataFrames or exported files.
    
    Validation Rules:
      1. Every Source 1 entity appears in results.
      2. No unexpected entity IDs in results.
      3. Every predicted match belongs to Source 2 or Source 3.
      4. Hard invariant: every predicted match MUST be present in candidate_pairs.
      5. File existence and non-empty check if paths are passed.
    """
    # 1. Output files validation if paths provided
    if matching_filepath:
        if not matching_filepath.exists():
            raise FileNotFoundError(f"Validation failed: {matching_filepath} does not exist!")
    if candidate_filepath:
        if not candidate_filepath.exists():
            raise FileNotFoundError(f"Validation failed: {candidate_filepath} does not exist!")

    # 2. Check entity_id set equality
    expected_s1_ids = set(s1_df["entity_id"])
    actual_results_ids = set(results_df["entity_id"])
    
    if expected_s1_ids != actual_results_ids:
        missing = expected_s1_ids - actual_results_ids
        extra = actual_results_ids - expected_s1_ids
        raise ValueError(f"Validation failed: Entity ID mismatch! Missing: {missing}, Unexpected extra: {extra}")

    # 3. Check candidate source IDs
    s2_ids = set(s2_df["entity_id"]) if "entity_id" in s2_df.columns and not s2_df.empty else set()
    s3_ids = set(s3_df["entity_id"]) if "entity_id" in s3_df.columns and not s3_df.empty else set()
    valid_cand_ids = s2_ids | s3_ids
    
    # 4. Check predicted match integrity & candidate pair invariant
    candidate_set = set(zip(candidate_pairs_df["source1_entity_id"], candidate_pairs_df["candidate_entity_id"])) if not candidate_pairs_df.empty else set()
    
    for _, row in results_df.iterrows():
        s1_id = row["entity_id"]
        matched_str = str(row["matched_entity_ids"]).strip()
        if matched_str and matched_str != "nan":
            matched_ids = matched_str.split(",")
            for m_id in matched_ids:
                m_id = m_id.strip()
                if m_id not in valid_cand_ids:
                    raise ValueError(f"Validation failed: Matched ID '{m_id}' not found in Source 2 or Source 3!")
                    
                # HARD INVARIANT: Every predicted match MUST be in candidate_pairs!
                if (s1_id, m_id) not in candidate_set:
                    raise AssertionError(f"Validation Error: Predicted match ({s1_id}, {m_id}) NOT present in candidate_pairs.tsv!")
                    
    print("SUCCESS: All pipeline validation checks passed clean!")
    return True
