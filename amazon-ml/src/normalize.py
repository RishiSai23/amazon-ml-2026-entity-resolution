"""Non-destructive text normalization and tokenization module."""

import re
import unicodedata
from typing import List, Set, Tuple
import pandas as pd

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase, unicode NFKD decomposition,
    replacing punctuation separators with spaces, and stripping extra whitespace.
    
    Args:
        text: Raw text string.
        
    Returns:
        Cleaned normalized string preserving essential alphanumeric tokens.
    """
    if not text or not isinstance(text, str):
        return ""
        
    # 1. Lowercase
    text = text.lower()
    
    # 2. Unicode normalization (NFKD to decompose accented characters)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    
    # 3. Replace common separators and punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    
    # 4. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def tokenize_text(text: str) -> List[str]:
    """Tokenize normalized text string into non-empty word tokens.
    
    Args:
        text: Normalized text string.
        
    Returns:
        List of string tokens.
    """
    if not text:
        return []
    return text.split()

def extract_legal_suffixes(tokens: List[str], legal_suffixes: Set[str]) -> Tuple[List[str], List[str]]:
    """Extract legal suffix tokens without deleting them from the original token stream.
    
    Args:
        tokens: Token list.
        legal_suffixes: Set of normalized legal suffixes.
        
    Returns:
        Tuple of (core_tokens, detected_suffixes)
    """
    found_suffixes = [t for t in tokens if t in legal_suffixes]
    core_tokens = [t for t in tokens if t not in legal_suffixes]
    return core_tokens, found_suffixes

def normalize_dataframe(df: pd.DataFrame, legal_suffixes: Set[str] = None) -> pd.DataFrame:
    """Apply non-destructive normalization to a DataFrame, creating explicit columns:
      - name_normalized
      - address_normalized
      - name_tokens
      - address_tokens
      - name_core_tokens
      - name_suffix_tokens
    
    Args:
        df: Input DataFrame containing 'business_name', 'business_address', 'country'.
        legal_suffixes: Set of legal suffixes for non-destructive tagging.
        
    Returns:
        DataFrame with new normalized and tokenized columns added.
    """
    if legal_suffixes is None:
        legal_suffixes = {"ltd", "limited", "pvt", "private", "inc", "corp", "corporation", "llc", "plc", "co", "company"}
        
    df = df.copy()
    
    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""
            
    df["name_normalized"] = df["business_name"].apply(normalize_text)
    df["address_normalized"] = df["business_address"].apply(normalize_text)
    df["country_normalized"] = df["country"].apply(normalize_text)
    
    df["name_tokens"] = df["name_normalized"].apply(tokenize_text)
    df["address_tokens"] = df["address_normalized"].apply(tokenize_text)
    
    # Extract core tokens vs suffix tokens non-destructively
    suffix_results = df["name_tokens"].apply(lambda tokens: extract_legal_suffixes(tokens, legal_suffixes))
    df["name_core_tokens"] = suffix_results.apply(lambda res: res[0])
    df["name_suffix_tokens"] = suffix_results.apply(lambda res: res[1])
    
    return df
