"""Data loading, validation, and profiling utilities for challenge TSV files."""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def load_tsv(file_path: Path, required_cols: List[str]) -> pd.DataFrame:
    """Load a TSV file with pandas and validate basic properties.
    
    Args:
        file_path: Absolute or relative Path object to TSV file.
        required_cols: List of expected column names.
        
    Returns:
        pd.DataFrame containing file data.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    df = pd.read_csv(file_path, sep="\t", dtype=str)
    
    # Fill NaN values with empty string for robust text processing
    for col in required_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = ""
            
    return df

def validate_dataframe(df: pd.DataFrame, source_name: str, required_cols: List[str]) -> None:
    """Validate required columns, non-empty state, and duplicate entity IDs.
    
    Args:
        df: Input DataFrame.
        source_name: Human readable identifier for dataset source.
        required_cols: Expected columns list.
    """
    if df.empty:
        raise ValueError(f"Dataset '{source_name}' is empty.")
        
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset '{source_name}' missing required columns: {missing_cols}")
        
    duplicate_ids = df[df.duplicated("entity_id", keep=False)]
    if not duplicate_ids.empty:
        dup_count = len(duplicate_ids["entity_id"].unique())
        raise ValueError(f"Dataset '{source_name}' contains {dup_count} duplicate entity_ids!")

def profile_dataset(df: pd.DataFrame, source_name: str) -> Dict[str, object]:
    """Profile dataset statistics without performing modeling.
    
    Args:
        df: Input DataFrame.
        source_name: Name of data source.
        
    Returns:
        Dict summarizing stats.
    """
    n_rows = len(df)
    n_unique_ids = df["entity_id"].nunique()
    missing_names = (df["business_name"] == "").sum()
    missing_addresses = (df["business_address"] == "").sum()
    missing_countries = (df["country"] == "").sum()
    n_unique_countries = df["country"].nunique()
    duplicate_records = df.duplicated(subset=["business_name", "business_address", "country"]).sum()
    
    stats = {
        "source_name": source_name,
        "number_of_rows": n_rows,
        "number_of_unique_entity_ids": n_unique_ids,
        "missing_business_names": missing_names,
        "missing_addresses": missing_addresses,
        "missing_countries": missing_countries,
        "number_of_unique_countries": n_unique_countries,
        "duplicate_records": duplicate_records,
    }
    
    print(f"\n--- Profile: {source_name} ---")
    for key, val in stats.items():
        if key != "source_name":
            print(f"  {key}: {val}")
    print("  Sample rows:")
    for _, row in df.head(2).iterrows():
        print(f"    [{row['entity_id']}] {row['business_name']} | {row['business_address']} | {row['country']}")
        
    return stats

def load_source_dataset(data_dir: Path, is_train: bool, required_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Source 1, Source 2, Source 3 and optional Ground Truth.
    
    Args:
        data_dir: Subdirectory containing files (train or test).
        is_train: Boolean flag indicating if ground truth should be loaded.
        required_cols: List of required columns.
        
    Returns:
        Tuple of (s1_df, s2_df, s3_df, gt_df)
    """
    prefix = "train" if is_train else "test"
    
    s1_path = data_dir / f"{prefix}_source1.tsv"
    s2_path = data_dir / f"{prefix}_source2.tsv"
    s3_path = data_dir / f"{prefix}_source3.tsv"
    
    s1_df = load_tsv(s1_path, required_cols)
    s2_df = load_tsv(s2_path, required_cols)
    s3_df = load_tsv(s3_path, required_cols)
    
    validate_dataframe(s1_df, f"{prefix}_source1", required_cols)
    validate_dataframe(s2_df, f"{prefix}_source2", required_cols)
    validate_dataframe(s3_df, f"{prefix}_source3", required_cols)
    
    gt_df = pd.DataFrame()
    if is_train:
        gt_path = data_dir / "train_ground_truth.tsv"
        if gt_path.exists():
            gt_df = load_tsv(gt_path, ["entity_id", "matched_entity_ids"])
            
    return s1_df, s2_df, s3_df, gt_df
