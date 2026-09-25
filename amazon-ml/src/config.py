"""Centralized configuration for the Business Entity Resolution pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

@dataclass
class PipelineConfig:
    """Pipeline settings for paths, normalization, blocking, features, and matching thresholds."""
    
    # Base Directories
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    dataset_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "output")
    
    # Mode-specific subdirectories
    train_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset" / "train")
    test_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset" / "test")
    
    # Required Columns
    required_columns: List[str] = field(default_factory=lambda: ["entity_id", "business_name", "business_address", "country"])
    
    # Legal Suffixes (for tag-based / non-destructive matching)
    legal_suffixes: Set[str] = field(default_factory=lambda: {
        "ltd", "limited", "pvt", "private", "inc", "incorporated", 
        "corp", "corporation", "llc", "co", "company", "plc", "gmbh", 
        "sa", "bv", "nv", "srl"
    })
    
    # Blocking Parameters
    min_token_len: int = 3
    char_sig_len: int = 3
    enable_exact_name_block: bool = True
    enable_name_prefix_block: bool = True
    enable_address_token_block: bool = True
    enable_char_signature_block: bool = True
    
    # Matching / Scoring Engine Parameters
    name_weight: float = 0.50
    address_weight: float = 0.35
    country_weight: float = 0.15
    match_threshold: float = 0.70
    
    # F0.5 evaluation metric weight beta
    beta_f05: float = 0.5
    
    def __post_init__(self):
        """Ensure paths exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
