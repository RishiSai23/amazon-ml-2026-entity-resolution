"""Non-destructive text normalization and tokenization module implementing N5 design."""

import re
import string
from typing import List, Set, Tuple
import pandas as pd
from anyascii import anyascii

PUNCT_TRANS_TABLE = str.maketrans(
    string.punctuation, " " * len(string.punctuation)
)

PUNCT_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

LEGAL_REPLACEMENTS_RAW = [
    (r"\bprivate limited\b", "pvt_ltd"),
    (r"\bpvt ltd\b", "pvt_ltd"),
    (r"\bpvt\b", "pvt_ltd"),
    (r"\bprivate\b", "pvt_ltd"),
    (r"\blimited\b", "ltd"),
    (r"\bltd\b", "ltd"),
    (r"\bincorporated\b", "inc"),
    (r"\binc\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcorp\b", "corp"),
    (r"\blimited liability company\b", "llc"),
    (r"\bllc\b", "llc"),
    (r"\blimited liability partnership\b", "llp"),
    (r"\bllp\b", "llp"),
    (r"\bsociete a responsabilite limitee\b", "sarl"),
    (r"\bs a r l\b", "sarl"),
    (r"\bsarl\b", "sarl"),
    (r"\bsociete anonyme\b", "sa"),
    (r"\bsa\b", "sa"),
    (r"\bcompany\b", "co"),
    (r"\bco\b", "co"),
]

ADDRESS_REPLACEMENTS_RAW = [
    (r"\broad\b", "road"), (r"\brd\b", "road"),
    (r"\bstreet\b", "street"), (r"\bst\b", "street"),
    (r"\bavenue\b", "avenue"), (r"\bave\b", "avenue"),
    (r"\bboulevard\b", "boulevard"), (r"\bblvd\b", "boulevard"),
    (r"\bdrive\b", "drive"), (r"\bdr\b", "drive"),
    (r"\blane\b", "lane"), (r"\bln\b", "lane"),
    (r"\bsuite\b", "suite"), (r"\bste\b", "suite"),
    (r"\bapartment\b", "apartment"), (r"\bapt\b", "apartment"),
    (r"\bbuilding\b", "building"), (r"\bbldg\b", "building"),
    (r"\bfloor\b", "floor"), (r"\bfl\b", "floor"),
    (r"\bnumber\b", "number"), (r"\bno\b", "number"), (r"\bnum\b", "number"),
    (r"\bpo box\b", "pobox"), (r"\bp o box\b", "pobox"), (r"\bpobox\b", "pobox")
]

LEGAL_REPLACEMENTS = [(re.compile(pat), rep) for pat, rep in LEGAL_REPLACEMENTS_RAW]
ADDRESS_REPLACEMENTS = [(re.compile(pat), rep) for pat, rep in ADDRESS_REPLACEMENTS_RAW]

CANONICAL_LEGAL_SUFFIXES = {
    "pvt_ltd", "ltd", "inc", "corp", "llc", "llp", "sarl", "sa", "co",
    "pvt", "private", "limited", "company", "corporation", "incorporated"
}

def normalize_text(text: str) -> str:
    """General baseline text normalization using lowercasing, anyascii transliteration,
    punctuation stripping, and whitespace collapse.
    """
    if not text or not isinstance(text, str):
        return ""
    text = anyascii(text)
    text = text.lower()
    text = PUNCT_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()

def normalize_name(text: str) -> str:
    """Normalize business name with N5 legal suffix canonicalization."""
    text = normalize_text(text)
    if not text:
        return ""
    for pat, rep in LEGAL_REPLACEMENTS:
        text = pat.sub(rep, text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()

def normalize_address(text: str) -> str:
    """Normalize business address with N5 address term canonicalization."""
    text = normalize_text(text)
    if not text:
        return ""
    for pat, rep in ADDRESS_REPLACEMENTS:
        text = pat.sub(rep, text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()

def tokenize_text(text: str) -> List[str]:
    """Tokenize normalized text string into non-empty word tokens."""
    if not text:
        return []
    return text.split()

def extract_legal_suffixes(tokens: List[str], legal_suffixes: Set[str]) -> Tuple[List[str], List[str]]:
    """Extract legal suffix tokens without deleting them from the original token stream."""
    if not tokens:
        return [], []
    found_suffixes = [t for t in tokens if t in legal_suffixes]
    core_tokens = [t for t in tokens if t not in legal_suffixes]
    return core_tokens, found_suffixes

def normalize_dataframe(df: pd.DataFrame, legal_suffixes: Set[str] = None) -> pd.DataFrame:
    """Apply non-destructive normalization to a DataFrame using fast vectorized operations."""
    target_suffixes = set(CANONICAL_LEGAL_SUFFIXES)
    if legal_suffixes:
        target_suffixes.update(legal_suffixes)
        
    df = df.copy()
    
    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""
            
    # Fast single-pass string normalization
    df["name_normalized"] = df["business_name"].fillna("").astype(str).apply(normalize_name)
    df["address_normalized"] = df["business_address"].fillna("").astype(str).apply(normalize_address)
    df["country_normalized"] = df["country"].fillna("").astype(str).apply(normalize_text)
    
    df["name_tokens"] = df["name_normalized"].str.split()
    df["address_tokens"] = df["address_normalized"].str.split()
    
    suffix_results = df["name_tokens"].apply(lambda tokens: extract_legal_suffixes(tokens, target_suffixes) if isinstance(tokens, list) else ([], []))
    df["name_core_tokens"] = suffix_results.apply(lambda res: res[0])
    df["name_suffix_tokens"] = suffix_results.apply(lambda res: res[1])
    
    return df
