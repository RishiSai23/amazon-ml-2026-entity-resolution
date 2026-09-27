"""Main CLI entry point executing end-to-end Entity Resolution pipeline."""

import argparse
import sys
from pathlib import Path
from src.streaming import run_streaming_test_pipeline
# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PipelineConfig
from src.io_utils import load_source_dataset, profile_dataset
from src.normalize import normalize_dataframe
from src.blocking import generate_candidate_pairs
from src.features import extract_pairwise_features
from src.matching import predict_matches, evaluate_predictions
from src.postprocess import format_matching_results, validate_pipeline_outputs
from src.generate_output import export_candidate_pairs, export_matching_results

def run_pipeline(mode: str = "mock", threshold: float = None) -> None:
    """Execute end-to-end Business Entity Resolution pipeline.
    
    Args:
        mode: Execution mode ('mock', 'train', or 'test').
        threshold: Optional matching score threshold override.
    """
    config = PipelineConfig()
    if threshold is not None:
        config.match_threshold = threshold
    
        # Test inference uses the RAM-safe streaming architecture.
    # mock/train intentionally retain the original validated pipeline.
    if mode == "test":
        run_streaming_test_pipeline(config)
        return
    print("=" * 65)
    print(f"  AMAZON ML CHALLENGE 2026 — ENTITY RESOLUTION PIPELINE")
    print(f"  Execution Mode: {mode.upper()} | Threshold: {config.match_threshold:.2f}")
    print("=" * 65)
    
    # 1. Determine directory and ground truth loading
    data_dir = config.train_dir if mode in ["mock", "train"] else config.test_dir
    is_train = mode in ["mock", "train"]
    
    # 2. Load data
    print("\n[Step 1/7] Loading Data...")
    s1_raw, s2_raw, s3_raw, gt_df = load_source_dataset(data_dir, is_train=is_train, required_cols=config.required_columns)
    
    # 3. Data Profiling
    print("\n[Step 2/7] Inspecting & Profiling Data...")
    profile_dataset(s1_raw, "Source 1 (Reference)")
    profile_dataset(s2_raw, "Source 2 (Noisy)")
    profile_dataset(s3_raw, "Source 3 (Noisy)")
    
    # 4. Text Normalization
    print("\n[Step 3/7] Normalizing & Tokenizing Text...")
    s1_norm = normalize_dataframe(s1_raw, config.legal_suffixes)
    s2_norm = normalize_dataframe(s2_raw, config.legal_suffixes)
    s3_norm = normalize_dataframe(s3_raw, config.legal_suffixes)
    
    # 5. Blocking / Candidate Generation
    print("\n[Step 4/7] Candidate Generation / Multi-Strategy Blocking...")
    candidate_pairs_df = generate_candidate_pairs(s1_norm, s2_norm, s3_norm, config)
    print(f"  Generated {len(candidate_pairs_df)} candidate pairs.")
    if not candidate_pairs_df.empty:
        print("  Sample blocking rules summary:")
        print(candidate_pairs_df["blocking_rules"].value_counts().to_string())
        
    # 6. Feature Extraction
    print("\n[Step 5/7] Pairwise Feature Engineering...")
    features_df = extract_pairwise_features(candidate_pairs_df, s1_norm, s2_norm, s3_norm)
    print(f"  Extracted {len(features_df.columns)} features for {len(features_df)} candidate pairs.")
    
    # 7. Matching Model & Predictions
    print("\n[Step 6/7] Baseline Weighted Matcher Decision...")
    predictions_df = predict_matches(features_df, config)
    print(f"  Predicted {len(predictions_df)} matches exceeding score threshold {config.match_threshold:.2f}.")
    
    # Evaluate against ground truth if available
    if is_train and not gt_df.empty:
        print("\n--- Training Ground Truth Evaluation ---")
        metrics = evaluate_predictions(candidate_pairs_df, predictions_df, gt_df, beta=config.beta_f05)
        print(f"  Candidate Count:   {metrics['candidate_count']}")
        print(f"  Candidate Recall:  {metrics['candidate_recall']:.4f}")
        print(f"  True Positives:    {metrics['true_positives']}")
        print(f"  False Positives:   {metrics['false_positives']}")
        print(f"  False Negatives:   {metrics['false_negatives']}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F0.5 Score:        {metrics['f0.5']:.4f}")
        print("---------------------------------------")
        
    # 8. Post-processing & Submission Export
    print("\n[Step 7/7] Post-processing & Output File Generation...")
    results_df = format_matching_results(s1_norm, predictions_df)
    
    cand_path = export_candidate_pairs(candidate_pairs_df, config.output_dir)
    res_path = export_matching_results(results_df, config.output_dir)
    
    # 9. Validation
    validate_pipeline_outputs(
        s1_norm, s2_norm, s3_norm,
        candidate_pairs_df, results_df,
        matching_filepath=res_path,
        candidate_filepath=cand_path
    )
    
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Amazon ML Challenge 2026 Entity Resolution Pipeline.")
    parser.add_argument("--mode", type=str, choices=["mock", "train", "test"], default="mock",
                        help="Execution mode: 'mock', 'train', or 'test' (default: mock)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional score threshold override for matcher (e.g. 0.75)")
    args = parser.parse_args()
    
    run_pipeline(mode=args.mode, threshold=args.threshold)
