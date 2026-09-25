"""Export writer module for matching_results.tsv and candidate_pairs.tsv."""

from pathlib import Path
import pandas as pd

def export_candidate_pairs(candidate_pairs_df: pd.DataFrame, output_dir: Path) -> Path:
    """Export candidate pairs to candidate_pairs.tsv in target directory.
    
    Args:
        candidate_pairs_df: DataFrame containing candidate pairs.
        output_dir: Path to output directory.
        
    Returns:
        Path to written file.
    """
    out_path = output_dir / "candidate_pairs.tsv"
    
    export_df = candidate_pairs_df.copy()
    if export_df.empty:
        export_df = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id", "candidate_source", "blocking_rules"
        ])
    else:
        export_df = export_df[["source1_entity_id", "candidate_entity_id", "candidate_source", "blocking_rules"]]
        
    export_df.to_csv(out_path, sep="\t", index=False)
    print(f"Exported candidate pairs to: {out_path} ({len(export_df)} rows)")
    return out_path

def export_matching_results(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Export final formatted matches to matching_results.tsv in target directory.
    
    Args:
        results_df: Formatted matching results DataFrame.
        output_dir: Path to output directory.
        
    Returns:
        Path to written file.
    """
    out_path = output_dir / "matching_results.tsv"
    
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"Exported matching results to: {out_path} ({len(results_df)} rows)")
    return out_path
