"""Lightweight Milestone 4 Experiment: 10,000 S1 Stratified Benchmark (N0 to N5 only).
Includes MAX_BLOCK_SIZE cap to prevent generic token explosion during evaluation.
"""

import time
import json
import string
import numpy as np
import pandas as pd
from anyascii import anyascii
from collections import defaultdict

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

MAX_BLOCK_SIZE = 5000

def fast_normalize_N0(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.translate(PUNCT_TRANS_TABLE).str.strip()

def fast_normalize_N1(series: pd.Series) -> pd.Series:
    return fast_normalize_N0(series)

def fast_normalize_N2(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.translate(ACCENT_PUNCT_TRANS_TABLE).str.strip()

def fast_normalize_N3(series: pd.Series) -> pd.Series:
    s = fast_normalize_N2(series)
    for pat, rep in LEGAL_REPLACEMENTS:
        s = s.str.replace(pat, rep, regex=True)
    return s.str.strip()

def fast_normalize_N4(series: pd.Series) -> pd.Series:
    s = fast_normalize_N3(series)
    for pat, rep in ADDRESS_REPLACEMENTS:
        s = s.str.replace(pat, rep, regex=True)
    return s.str.strip()

def fast_normalize_N5(series: pd.Series) -> pd.Series:
    ascii_s = series.fillna("").astype(str).apply(lambda text: anyascii(text))
    return fast_normalize_N4(ascii_s)

def get_char_signature(name_norm: str, sig_len: int = 3) -> str:
    clean_chars = [c for c in name_norm if c.isalnum()]
    if not clean_chars:
        return ""
    return "".join(sorted(clean_chars[:sig_len]))

def main():
    print("======================================================================", flush=True)
    print("LIGHTWEIGHT MILESTONE 4 EXPERIMENT (10,000 S1 BENCHMARK)", flush=True)
    print("======================================================================", flush=True)
    
    t0 = time.time()
    train_dir = "dataset/train/"
    
    print("[Step 1] Loading Datasets...", flush=True)
    s1_df = pd.read_csv(train_dir + "train_source1.tsv", sep="\t")
    s2_df = pd.read_csv(train_dir + "train_source2.tsv", sep="\t")
    s3_df = pd.read_csv(train_dir + "train_source3.tsv", sep="\t")
    gt_df = pd.read_csv(train_dir + "train_ground_truth.tsv", sep="\t")
    
    print("Parsing Ground Truth map...", flush=True)
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
    
    benchmark_size = 10000
    print(f"Sampling {benchmark_size:,} S1 entities stratified by Country & Match Cardinality (seed=42)...", flush=True)
    np.random.seed(42)
    
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
    
    norm_fn_dict = {
        "N0": (fast_normalize_N0, "Baseline (N0)"),
        "N1": (fast_normalize_N1, "Unicode & Punctuation (N1)"),
        "N2": (fast_normalize_N2, "Accent Stripping (N2)"),
        "N3": (fast_normalize_N3, "Legal Suffix Canonicalization (N3)"),
        "N4": (fast_normalize_N4, "Address Abbreviation Canonicalization (N4)"),
        "N5": (fast_normalize_N5, "Transliteration (N5)"),
    }
    
    results = []
    
    for norm_key, (norm_fn, label) in norm_fn_dict.items():
        print(f"\n--- Evaluating {label} ---", flush=True)
        t_var = time.time()
        
        s1_prep = s1_sample_df.copy()
        s1_prep["name_norm"] = norm_fn(s1_prep["business_name"])
        s1_prep["addr_norm"] = norm_fn(s1_prep["business_address"])
        s1_prep["country_norm"] = fast_normalize_N0(s1_prep["country"])
        s1_prep["char_sig"] = s1_prep["name_norm"].apply(get_char_signature)
        
        s2_prep = s2_df.copy()
        s2_prep["name_norm"] = norm_fn(s2_prep["business_name"])
        s2_prep["addr_norm"] = norm_fn(s2_prep["business_address"])
        s2_prep["country_norm"] = fast_normalize_N0(s2_prep["country"])
        s2_prep["char_sig"] = s2_prep["name_norm"].apply(get_char_signature)
        
        s3_prep = s3_df.copy()
        s3_prep["name_norm"] = norm_fn(s3_prep["business_name"])
        s3_prep["addr_norm"] = norm_fn(s3_prep["business_address"])
        s3_prep["country_norm"] = fast_normalize_N0(s3_prep["country"])
        s3_prep["char_sig"] = s3_prep["name_norm"].apply(get_char_signature)
        
        exact_idx = defaultdict(set)
        ntoken_idx = defaultdict(set)
        atoken_idx = defaultdict(set)
        csig_idx = defaultdict(set)
        
        def index_df(cand_df):
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
                    for t in name_norm.split():
                        if len(t) >= 3:
                            ntoken_idx[(country, t)].add(cand)
                if addr_norm:
                    for t in addr_norm.split():
                        if t.isdigit() or len(t) >= 4:
                            atoken_idx[(country, t)].add(cand)
                if sig:
                    csig_idx[(country, sig)].add(cand)
                    
        index_df(s2_prep)
        index_df(s3_prep)
        
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
            
            if name_norm:
                c_set = exact_idx.get((country, name_norm), set())
                if len(c_set) <= MAX_BLOCK_SIZE:
                    cands_A.update(c_set)
                for t in name_norm.split():
                    if len(t) >= 3:
                        t_set = ntoken_idx.get((country, t), set())
                        if len(t_set) <= MAX_BLOCK_SIZE:
                            cands_B.update(t_set)
                        
            if addr_norm:
                for t in addr_norm.split():
                    if t.isdigit() or len(t) >= 4:
                        t_set = atoken_idx.get((country, t), set())
                        if len(t_set) <= MAX_BLOCK_SIZE:
                            cands_C.update(t_set)
                        
            if sig:
                s_set = csig_idx.get((country, sig), set())
                if len(s_set) <= MAX_BLOCK_SIZE:
                    cands_D.update(s_set)
                
            union_cands = cands_A | cands_B | cands_C | cands_D
            candidate_counts.append(len(union_cands))
            
            # O(1) GT lookup
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
        rec_A = captured_A / total_gt
        rec_B = captured_B / total_gt
        rec_C = captured_C / total_gt
        rec_D = captured_D / total_gt
        rec_E = captured_E / total_gt
        
        counts_arr = np.array(candidate_counts)
        mean_cands = float(np.mean(counts_arr))
        median_cands = float(np.median(counts_arr))
        p95_cands = float(np.percentile(counts_arr, 95))
        p99_cands = float(np.percentile(counts_arr, 99))
        
        elapsed = time.time() - t_var
        
        res = {
            "variant": norm_key,
            "label": label,
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
            "runtime_seconds": elapsed
        }
        results.append(res)
        
        print(f"  Exact Name Recall:    {rec_A*100:.2f}%")
        print(f"  Name Token Recall:    {rec_B*100:.2f}%")
        print(f"  Address Token Recall: {rec_C*100:.2f}%")
        print(f"  Char Sig Recall:      {rec_D*100:.2f}%")
        print(f"  UNION RECALL:         {rec_E*100:.2f}% ({captured_E:,}/{total_gt:,})")
        print(f"  Candidates/S1: Mean={mean_cands:.1f}, Med={median_cands:.0f}, P95={p95_cands:.0f}, P99={p99_cands:.0f}")
        print(f"  Runtime: {elapsed:.2f}s", flush=True)

    output_payload = {
        "benchmark_sample_size": benchmark_size,
        "sample_gt_pairs_count": len(sample_gt_pairs),
        "total_runtime_seconds": time.time() - t0,
        "normalization_results": results
    }
    
    with open("scratch/milestone_4_final_results.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
        
    print(f"\n======================================================================")
    print(f"FINAL LIGHTWEIGHT EXPERIMENT COMPLETE in {time.time() - t0:.2f}s")
    print(f"Results saved to scratch/milestone_4_final_results.json")
    print(f"======================================================================", flush=True)

if __name__ == "__main__":
    main()
