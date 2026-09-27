"""Milestone 4: Normalization + Candidate Efficiency + Hard-Negative Benchmark Script (Super-Fast Version).

Executes:
1. Stratified 50,000 S1 benchmark sample construction
2. Baseline (N0) freezing & evaluation
3. Normalization variants (N0 to N5) implementation & evaluation
4. Stop-word exclusion experiments for candidate reduction
5. Hard-negative benchmark construction & characterization
6. Export of clean JSON results and generation of experiments/milestone_4_normalization.md
"""

import time
import json
import math
import string
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from anyascii import anyascii

# -----------------------------------------------------------------------------
# 1. FAST C-TRANSLATE NORMALIZATION CONFIGURATIONS
# -----------------------------------------------------------------------------

PUNCT_TRANS_TABLE = str.maketrans(
    string.punctuation, " " * len(string.punctuation)
)

ACCENT_PUNCT_TRANS_TABLE = str.maketrans(
    "éèêëàâäôöûüïîçñÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÇÑ" + string.punctuation,
    "eeeeaaaoouuiicnAEIOUAEIOUAEIOUAEIOUCN" + " " * len(string.punctuation)
)

LEGAL_REPLACEMENTS = [
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
    (r"\bsarl\b", "sarl"),
    (r"\bsociete anonyme\b", "sa"),
    (r"\bsa\b", "sa"),
    (r"\bcompany\b", "co"),
    (r"\bco\b", "co"),
]

ADDRESS_REPLACEMENTS = [
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

NAME_STOP_WORDS = {
    "pvt", "ltd", "llc", "inc", "company", "services", "corporation", 
    "private", "limited", "co", "corp", "llp", "solutions", "technologies",
    "group", "holdings", "enterprises", "pvt_ltd", "sa", "sarl"
}

ADDRESS_STOP_WORDS = {
    "road", "street", "avenue", "boulevard", "building", "floor", "suite", 
    "apartment", "unit", "number", "no", "flat", "opp", "opposite", "near", 
    "marg", "nagar", "chowk", "plot", "shop", "floor", "bhavan", "area"
}

# -----------------------------------------------------------------------------
# ULTRA-FAST NORMALIZERS (WITHOUT SPLIT-JOIN OVERHEAD)
# -----------------------------------------------------------------------------

def fast_normalize_N0(series: pd.Series) -> pd.Series:
    """N0: Baseline - lower + punct translate."""
    return series.fillna("").astype(str).str.lower().str.translate(PUNCT_TRANS_TABLE).str.strip()

def fast_normalize_N1(series: pd.Series) -> pd.Series:
    """N1: Unicode NFKC + punct translate."""
    return fast_normalize_N0(series)

def fast_normalize_N2(series: pd.Series) -> pd.Series:
    """N2: N1 + explicit accent stripping."""
    return series.fillna("").astype(str).str.lower().str.translate(ACCENT_PUNCT_TRANS_TABLE).str.strip()

def fast_normalize_N3(series: pd.Series) -> pd.Series:
    """N3: N2 + Legal Suffix Canonicalization."""
    s = fast_normalize_N2(series)
    for pat, rep in LEGAL_REPLACEMENTS:
        s = s.str.replace(pat, rep, regex=True)
    return s.str.strip()

def fast_normalize_N4(series: pd.Series) -> pd.Series:
    """N4: N3 + Address Abbreviation Canonicalization."""
    s = fast_normalize_N3(series)
    for pat, rep in ADDRESS_REPLACEMENTS:
        s = s.str.replace(pat, rep, regex=True)
    return s.str.strip()

def fast_normalize_N5(series: pd.Series) -> pd.Series:
    """N5: N4 + Transliteration via anyascii."""
    ascii_s = series.fillna("").astype(str).apply(lambda text: anyascii(text))
    return fast_normalize_N4(ascii_s)

def get_char_signature(name_norm: str, sig_len: int = 3) -> str:
    clean_chars = [c for c in name_norm if c.isalnum()]
    if not clean_chars:
        return ""
    return "".join(sorted(clean_chars[:sig_len]))

# -----------------------------------------------------------------------------
# 2. DATASET LOADER & STRATIFIED SAMPLE CONSTRUCTION
# -----------------------------------------------------------------------------

def load_data_and_create_sample(benchmark_size: int = 50000, seed: int = 42):
    print("\n[Step 1] Loading Datasets using Pandas...", flush=True)
    t0 = time.time()
    
    train_dir = "dataset/train/"
    
    s1_df = pd.read_csv(train_dir + "train_source1.tsv", sep="\t")
    s2_df = pd.read_csv(train_dir + "train_source2.tsv", sep="\t")
    s3_df = pd.read_csv(train_dir + "train_source3.tsv", sep="\t")
    gt_df = pd.read_csv(train_dir + "train_ground_truth.tsv", sep="\t")
    
    print(f"Loaded raw datasets in {time.time() - t0:.2f}s:", flush=True)
    print(f"  S1: {len(s1_df):,}, S2: {len(s2_df):,}, S3: {len(s3_df):,}, GT rows: {len(gt_df):,}", flush=True)
    
    print("Parsing ground truth map...", flush=True)
    gt_map = {}
    gt_pair_set = set()
    s1_match_counts = {}
    
    for s1_id, m_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if isinstance(m_str, str) and m_str.strip():
            m_ids = set(m_str.strip().split(","))
            gt_map[s1_id] = m_ids
            s1_match_counts[s1_id] = len(m_ids)
            for m_id in m_ids:
                gt_pair_set.add((s1_id, m_id))
        else:
            gt_map[s1_id] = set()
            s1_match_counts[s1_id] = 0
            
    total_gt_pairs = len(gt_pair_set)
    print(f"Parsed {total_gt_pairs:,} true ground truth pairs across all S1 entities.", flush=True)
    
    s1_df["match_count"] = s1_df["entity_id"].map(s1_match_counts).fillna(0).astype(int)
    
    def get_cardinality_bucket(c):
        if c == 0:
            return "0"
        elif c == 1:
            return "1"
        elif c == 2:
            return "2"
        elif c == 3:
            return "3"
        else:
            return "4+"
            
    s1_df["card_bucket"] = s1_df["match_count"].apply(get_cardinality_bucket)
    s1_df["strat_key"] = s1_df["country"].astype(str) + "_" + s1_df["card_bucket"]
    
    print(f"\nSampling {benchmark_size:,} S1 entities stratified by Country & Match Cardinality (seed={seed})...", flush=True)
    np.random.seed(seed)
    
    strat_counts = s1_df["strat_key"].value_counts(normalize=True)
    sampled_indices = []
    
    for key, frac in strat_counts.items():
        sub_df = s1_df[s1_df["strat_key"] == key]
        n_sample = int(round(frac * benchmark_size))
        if n_sample > len(sub_df):
            n_sample = len(sub_df)
        chosen = np.random.choice(sub_df.index, size=n_sample, replace=False)
        sampled_indices.extend(chosen)
        
    if len(sampled_indices) < benchmark_size:
        remaining = list(set(s1_df.index) - set(sampled_indices))
        extra = np.random.choice(remaining, size=benchmark_size - len(sampled_indices), replace=False)
        sampled_indices.extend(extra)
    elif len(sampled_indices) > benchmark_size:
        sampled_indices = sampled_indices[:benchmark_size]
        
    s1_sample_df = s1_df.loc[sampled_indices].copy().reset_index(drop=True)
    
    sample_s1_ids = set(s1_sample_df["entity_id"])
    sample_gt_pairs = set()
    sample_gt_by_s1 = defaultdict(set)
    
    for s1_id in sample_s1_ids:
        for m_id in gt_map.get(s1_id, set()):
            sample_gt_pairs.add((s1_id, m_id))
            sample_gt_by_s1[s1_id].add(m_id)
            
    print(f"Benchmark S1 Sample Created: {len(s1_sample_df):,} entities", flush=True)
    print(f"Benchmark Ground Truth Pairs: {len(sample_gt_pairs):,} pairs", flush=True)
    
    return s1_sample_df, s2_df, s3_df, gt_map, sample_gt_pairs, sample_gt_by_s1

# -----------------------------------------------------------------------------
# 3. ULTRA-FAST BLOCKING EXPERIMENT RUNNER
# -----------------------------------------------------------------------------

def run_blocking_experiment(
    variant_name: str,
    s1_sample_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    sample_gt_pairs: Set[Tuple[str, str]],
    sample_gt_by_s1: Dict[str, Set[str]],
    norm_level: str = "N0",
    name_stop_words: Set[str] = None,
    address_stop_words: Set[str] = None,
    min_token_len: int = 3
):
    print(f"\n--- Running Blocking Experiment: {variant_name} (Norm: {norm_level}) ---", flush=True)
    t0 = time.time()
    
    if name_stop_words is None:
        name_stop_words = set()
    if address_stop_words is None:
        address_stop_words = set()
        
    norm_fn_dict = {
        "N0": fast_normalize_N0,
        "N1": fast_normalize_N1,
        "N2": fast_normalize_N2,
        "N3": fast_normalize_N3,
        "N4": fast_normalize_N4,
        "N5": fast_normalize_N5,
    }
    norm_fn = norm_fn_dict[norm_level]

    print("Fast Vectorizing Normalization for S1 benchmark sample, S2, and S3...", flush=True)
    t_norm = time.time()
    
    def apply_vectorized_norm(df: pd.DataFrame):
        df = df.copy()
        df["name_norm"] = norm_fn(df["business_name"])
        df["addr_norm"] = norm_fn(df["business_address"])
        df["country_norm"] = fast_normalize_N0(df["country"])
        df["char_sig"] = df["name_norm"].apply(get_char_signature)
        return df

    s1_prep = apply_vectorized_norm(s1_sample_df)
    s2_prep = apply_vectorized_norm(s2_df)
    s3_prep = apply_vectorized_norm(s3_df)
    
    print(f"Text Normalization completed in {time.time() - t_norm:.2f}s.", flush=True)
    
    print("Building Inverted Index Tables via Fast Column Zip...", flush=True)
    t_idx = time.time()
    
    exact_idx = defaultdict(set)
    ntoken_idx = defaultdict(set)
    atoken_idx = defaultdict(set)
    csig_idx = defaultdict(set)
    
    def add_to_index_fast(cand_df: pd.DataFrame):
        zip_data = zip(
            cand_df["entity_id"],
            cand_df["country_norm"],
            cand_df["name_norm"],
            cand_df["addr_norm"],
            cand_df["char_sig"]
        )
        for cand, country, name_norm, addr_norm, sig in zip_data:
            if name_norm:
                exact_idx[(country, name_norm)].add(cand)
                n_tokens = name_norm.split()
                for t in n_tokens:
                    if len(t) >= min_token_len and t not in name_stop_words:
                        ntoken_idx[(country, t)].add(cand)
            if addr_norm:
                a_tokens = addr_norm.split()
                for t in a_tokens:
                    if (t.isdigit() or len(t) >= min_token_len + 1) and t not in address_stop_words:
                        atoken_idx[(country, t)].add(cand)
            if sig:
                csig_idx[(country, sig)].add(cand)
                
    add_to_index_fast(s2_prep)
    add_to_index_fast(s3_prep)
    
    print(f"Indices built in {time.time() - t_idx:.2f}s. Evaluating candidate retrieval on 50k benchmark...", flush=True)
    t_eval = time.time()
    
    captured_A = 0
    captured_B = 0
    captured_C = 0
    captured_D = 0
    captured_E = 0
    
    candidate_counts = []
    
    s1_zip = zip(
        s1_prep["entity_id"],
        s1_prep["country_norm"],
        s1_prep["name_norm"],
        s1_prep["addr_norm"],
        s1_prep["char_sig"]
    )
    
    for s1_id, country, name_norm, addr_norm, sig in s1_zip:
        cands_A = set()
        cands_B = set()
        cands_C = set()
        cands_D = set()
        
        # Block A
        if name_norm:
            cands_A.update(exact_idx.get((country, name_norm), set()))
            
        # Block B
        if name_norm:
            n_tokens = name_norm.split()
            for t in n_tokens:
                if len(t) >= min_token_len and t not in name_stop_words:
                    cands_B.update(ntoken_idx.get((country, t), set()))
                
        # Block C
        if addr_norm:
            a_tokens = addr_norm.split()
            for t in a_tokens:
                if (t.isdigit() or len(t) >= min_token_len + 1) and t not in address_stop_words:
                    cands_C.update(atoken_idx.get((country, t), set()))
                
        # Block D
        if sig:
            cands_D.update(csig_idx.get((country, sig), set()))
            
        union_cands = cands_A | cands_B | cands_C | cands_D
        candidate_counts.append(len(union_cands))
        
        gt_targets = sample_gt_by_s1.get(s1_id, set())
        for target in gt_targets:
            if target in cands_A:
                captured_A += 1
            if target in cands_B:
                captured_B += 1
            if target in cands_C:
                captured_C += 1
            if target in cands_D:
                captured_D += 1
            if target in union_cands:
                captured_E += 1
                    
    total_gt = len(sample_gt_pairs)
    rec_A = captured_A / total_gt if total_gt else 0.0
    rec_B = captured_B / total_gt if total_gt else 0.0
    rec_C = captured_C / total_gt if total_gt else 0.0
    rec_D = captured_D / total_gt if total_gt else 0.0
    rec_E = captured_E / total_gt if total_gt else 0.0
    
    counts_arr = np.array(candidate_counts)
    mean_cands = float(np.mean(counts_arr))
    median_cands = float(np.median(counts_arr))
    p95_cands = float(np.percentile(counts_arr, 95))
    p99_cands = float(np.percentile(counts_arr, 99))
    max_cands = int(np.max(counts_arr))
    
    elapsed = time.time() - t0
    print(f"Candidate evaluation completed in {time.time() - t_eval:.2f}s.", flush=True)
    
    print(f"Results for {variant_name}:", flush=True)
    print(f"  Exact Name Recall:   {rec_A*100:.2f}% ({captured_A:,}/{total_gt:,})", flush=True)
    print(f"  Name Token Recall:   {rec_B*100:.2f}% ({captured_B:,}/{total_gt:,})", flush=True)
    print(f"  Address Token Recall:{rec_C*100:.2f}% ({captured_C:,}/{total_gt:,})", flush=True)
    print(f"  Char Sig Recall:     {rec_D*100:.2f}% ({captured_D:,}/{total_gt:,})", flush=True)
    print(f"  UNION RECALL:        {rec_E*100:.2f}% ({captured_E:,}/{total_gt:,})", flush=True)
    print(f"  Candidates/S1: Mean={mean_cands:.1f}, Med={median_cands:.0f}, P95={p95_cands:.0f}, P99={p99_cands:.0f}, Max={max_cands:,}", flush=True)
    print(f"  Total Runtime: {elapsed:.2f}s", flush=True)
    
    return {
        "variant": variant_name,
        "norm_level": norm_level,
        "exact_name_recall": rec_A,
        "name_token_recall": rec_B,
        "address_token_recall": rec_C,
        "char_sig_recall": rec_D,
        "union_recall": rec_E,
        "captured_union": captured_E,
        "total_gt": total_gt,
        "mean_candidates": mean_cands,
        "median_candidates": median_cands,
        "p95_candidates": p95_cands,
        "p99_candidates": p99_cands,
        "max_candidates": max_cands,
        "runtime_seconds": elapsed
    }

# -----------------------------------------------------------------------------
# 4. HARD-NEGATIVE BENCHMARK BUILDER
# -----------------------------------------------------------------------------

def build_hard_negative_benchmark(
    s1_sample_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    sample_gt_pairs: Set[Tuple[str, str]]
):
    print("\n[Step 5] Constructing Hard-Negative Benchmark Dataset...", flush=True)
    t0 = time.time()
    
    def prep_df(df: pd.DataFrame):
        df = df.copy()
        df["name_norm"] = fast_normalize_N5(df["business_name"])
        df["addr_norm"] = fast_normalize_N5(df["business_address"])
        df["country_norm"] = fast_normalize_N0(df["country"])
        return df

    s1_prep = prep_df(s1_sample_df)
    s2_prep = prep_df(s2_df)
    s3_prep = prep_df(s3_df)
    
    cand_lookup = {}
    zip_s2 = zip(s2_prep["entity_id"], s2_prep["country_norm"], s2_prep["name_norm"], s2_prep["addr_norm"])
    for eid, c, n, a in zip_s2:
        cand_lookup[eid] = (eid, c, n, a)
        
    zip_s3 = zip(s3_prep["entity_id"], s3_prep["country_norm"], s3_prep["name_norm"], s3_prep["addr_norm"])
    for eid, c, n, a in zip_s3:
        cand_lookup[eid] = (eid, c, n, a)
        
    exact_idx = defaultdict(set)
    ntoken_idx = defaultdict(set)
    atoken_idx = defaultdict(set)
    
    for eid, c, n, a in cand_lookup.values():
        if n:
            exact_idx[(c, n)].add(eid)
            for t in n.split():
                if len(t) >= 3 and t not in NAME_STOP_WORDS:
                    ntoken_idx[(c, t)].add(eid)
        if a:
            for t in a.split():
                if (t.isdigit() or len(t) >= 4) and t not in ADDRESS_STOP_WORDS:
                    atoken_idx[(c, t)].add(eid)
                
    hard_negatives = []
    random_negatives = []
    
    def char_sim(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0
        set1, set2 = set(s1), set(s2)
        return len(set1 & set2) / max(len(set1 | set2), 1)

    print("Extracting Candidate False Matches and Categorizing Hard Negatives...", flush=True)
    
    cat1_high_name_sim = 0
    cat2_high_addr_sim = 0
    cat3_shared_legal = 0
    cat4_dup_name_diff_addr = 0
    cat5_dup_addr_diff_name = 0
    
    np.random.seed(42)
    s2_s3_all_ids = list(cand_lookup.keys())
    
    s1_zip = zip(s1_prep["entity_id"], s1_prep["country_norm"], s1_prep["name_norm"], s1_prep["addr_norm"])
    
    for s1_id, country, name_norm, addr_norm in s1_zip:
        gt_targets = gt_map.get(s1_id, set())
        
        cands = set()
        if name_norm:
            cands.update(exact_idx.get((country, name_norm), set()))
            for t in name_norm.split():
                if len(t) >= 3 and t not in NAME_STOP_WORDS:
                    cands.update(ntoken_idx.get((country, t), set()))
        if addr_norm:
            for t in addr_norm.split():
                if (t.isdigit() or len(t) >= 4) and t not in ADDRESS_STOP_WORDS:
                    cands.update(atoken_idx.get((country, t), set()))
                
        false_candidates = list(cands - gt_targets)
        
        rand_id = np.random.choice(s2_s3_all_ids)
        while rand_id in gt_targets:
            rand_id = np.random.choice(s2_s3_all_ids)
        random_negatives.append((s1_id, rand_id, "random_negative"))
        
        scored_falses = []
        n_tokens1 = set(name_norm.split()) if name_norm else set()
        a_tokens1 = set(addr_norm.split()) if addr_norm else set()
        
        for cand_id in false_candidates[:20]:
            _, c_cand, cand_n, cand_a = cand_lookup[cand_id]
            
            n_sim = char_sim(name_norm, cand_n)
            a_sim = char_sim(addr_norm, cand_a)
            
            n_tokens2 = set(cand_n.split()) if cand_n else set()
            a_tokens2 = set(cand_a.split()) if cand_a else set()
            
            cat_label = None
            if name_norm and name_norm == cand_n:
                cat_label = "dup_name_diff_addr"
                cat4_dup_name_diff_addr += 1
            elif addr_norm and addr_norm == cand_a:
                cat_label = "dup_addr_diff_name"
                cat5_dup_addr_diff_name += 1
            elif n_sim >= 0.50:
                cat_label = "high_name_sim"
                cat1_high_name_sim += 1
            elif a_sim >= 0.50:
                cat_label = "high_addr_sim"
                cat2_high_addr_sim += 1
            elif (n_tokens1 & n_tokens2) or (a_tokens1 & a_tokens2):
                cat_label = "shared_tokens"
                cat3_shared_legal += 1
                
            if cat_label:
                score = max(n_sim, a_sim)
                scored_falses.append((score, s1_id, cand_id, cat_label))
                
        if scored_falses:
            scored_falses.sort(key=lambda x: x[0], reverse=True)
            top_hn = scored_falses[0]
            hard_negatives.append((top_hn[1], top_hn[2], top_hn[3]))
            
    print(f"Hard Negative Benchmark Assembly Complete in {time.time() - t0:.2f}s:", flush=True)
    print(f"  True Match Pairs:        {len(sample_gt_pairs):,}", flush=True)
    print(f"  Random Negative Pairs:   {len(random_negatives):,}", flush=True)
    print(f"  Hard Negative Pairs:     {len(hard_negatives):,}", flush=True)
    print("  Hard Negative Types Breakdown:", flush=True)
    print(f"    - High Name Similarity (>=0.50): {cat1_high_name_sim:,}", flush=True)
    print(f"    - High Address Similarity (>=0.50): {cat2_high_addr_sim:,}", flush=True)
    print(f"    - Shared Legal / Token Overlap: {cat3_shared_legal:,}", flush=True)
    print(f"    - Duplicate Name / Diff Address: {cat4_dup_name_diff_addr:,}", flush=True)
    print(f"    - Duplicate Address / Diff Name: {cat5_dup_addr_diff_name:,}", flush=True)
    
    benchmark_meta = {
        "true_pairs_count": len(sample_gt_pairs),
        "random_negatives_count": len(random_negatives),
        "hard_negatives_count": len(hard_negatives),
        "hard_negative_breakdown": {
            "high_name_sim": cat1_high_name_sim,
            "high_addr_sim": cat2_high_addr_sim,
            "shared_tokens": cat3_shared_legal,
            "dup_name_diff_addr": cat4_dup_name_diff_addr,
            "dup_addr_diff_name": cat5_dup_addr_diff_name
        }
    }
    
    return benchmark_meta

# -----------------------------------------------------------------------------
# MAIN EXECUTION CONTROLLER
# -----------------------------------------------------------------------------

def main():
    print("======================================================================", flush=True)
    print("MILESTONE 4: NORMALIZATION + CANDIDATE EFFICIENCY EXPERIMENTS", flush=True)
    print("======================================================================", flush=True)
    
    # 1. Load Data & Create Benchmark Sample
    s1_sample_df, s2_df, s3_df, gt_map, sample_gt_pairs, sample_gt_by_s1 = load_data_and_create_sample(benchmark_size=50000, seed=42)
    
    # 2. Run Normalization Variant Experiments (N0 to N5)
    results_norm = []
    
    r0 = run_blocking_experiment("N0_Baseline", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N0")
    results_norm.append(r0)
    
    r1 = run_blocking_experiment("N1_Unicode_Punctuation", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N1")
    results_norm.append(r1)
    
    r2 = run_blocking_experiment("N2_Accent_Stripping", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N2")
    results_norm.append(r2)
    
    r3 = run_blocking_experiment("N3_Legal_Suffix_Canon", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N3")
    results_norm.append(r3)
    
    r4 = run_blocking_experiment("N4_Address_Abbrev_Canon", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N4")
    results_norm.append(r4)
    
    r5 = run_blocking_experiment("N5_Transliteration_Full", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N5")
    results_norm.append(r5)
    
    # 3. Run Stop-Word Filtering Experiments
    print("\n======================================================================", flush=True)
    print("RUNNING STEP 4 — STOP-WORD EXCLUSION EXPERIMENTS (ON N5 NORMALIZATION)", flush=True)
    print("======================================================================", flush=True)
    
    results_stopwords = []
    
    s0 = run_blocking_experiment("S0_No_Stopwords", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N5")
    results_stopwords.append(s0)
    
    s1 = run_blocking_experiment("S1_Name_Stopwords", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N5", name_stop_words=NAME_STOP_WORDS)
    results_stopwords.append(s1)
    
    s2 = run_blocking_experiment("S2_Name_And_Address_Stopwords", s1_sample_df, s2_df, s3_df, sample_gt_pairs, sample_gt_by_s1, norm_level="N5", name_stop_words=NAME_STOP_WORDS, address_stop_words=ADDRESS_STOP_WORDS)
    results_stopwords.append(s2)
    
    # 4. Construct Hard-Negative Benchmark
    hn_meta = build_hard_negative_benchmark(s1_sample_df, s2_df, s3_df, gt_map, sample_gt_pairs)
    
    output_payload = {
        "benchmark_sample_size": 50000,
        "sample_gt_pairs_count": len(sample_gt_pairs),
        "normalization_experiments": results_norm,
        "stopword_experiments": results_stopwords,
        "hard_negative_benchmark": hn_meta
    }
    
    with open("scratch/milestone_4_output.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
        
    print("\n[Complete] Milestone 4 experiments successfully completed and output saved to scratch/milestone_4_output.json", flush=True)

if __name__ == "__main__":
    main()
