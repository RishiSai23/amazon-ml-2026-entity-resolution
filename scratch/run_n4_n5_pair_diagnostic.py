"""Pair-Level Diagnostic for Normalization Variants N4 and N5.
Evaluates N3, N4, and N5 normalization on all 34,691 true ground-truth pairs 
from the 10,000 S1 stratified benchmark sample.
"""

import time
import json
import string
import numpy as np
import pandas as pd
from anyascii import anyascii
import difflib

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

def fast_normalize_N2_str(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower().translate(ACCENT_PUNCT_TRANS_TABLE).strip()

def fast_normalize_N3(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower().str.translate(ACCENT_PUNCT_TRANS_TABLE).str.strip()
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

def token_jaccard(s1: str, s2: str) -> float:
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def char_sim(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def main():
    print("======================================================================", flush=True)
    print("LIGHTWEIGHT PAIR-LEVEL DIAGNOSTIC FOR N4 & N5", flush=True)
    print("======================================================================", flush=True)

    t0 = time.time()
    train_dir = "dataset/train/"
    
    print("[1/4] Loading Datasets and Ground Truth...", flush=True)
    s1_df = pd.read_csv(train_dir + "train_source1.tsv", sep="\t")
    s2_df = pd.read_csv(train_dir + "train_source2.tsv", sep="\t")
    s3_df = pd.read_csv(train_dir + "train_source3.tsv", sep="\t")
    gt_df = pd.read_csv(train_dir + "train_ground_truth.tsv", sep="\t")

    gt_map = {}
    s1_match_counts = {}
    for s1_id, m_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if isinstance(m_str, str) and m_str.strip():
            m_ids = set(m_str.strip().split(","))
            gt_map[s1_id] = m_ids
            s1_match_counts[s1_id] = len(m_ids)
        else:
            gt_map[s1_id] = set()
            s1_match_counts[s1_id] = 0

    s1_df["match_count"] = s1_df["entity_id"].map(s1_match_counts).fillna(0).astype(int)

    def get_cardinality_bucket(c):
        if c == 0: return "0"
        elif c == 1: return "1"
        elif c == 2: return "2"
        elif c == 3: return "3"
        else: return "4+"

    s1_df["card_bucket"] = s1_df["match_count"].apply(get_cardinality_bucket)
    s1_df["strat_key"] = s1_df["country"].astype(str) + "_" + s1_df["card_bucket"]

    # Stratified 10k S1 sample with seed=42
    benchmark_size = 10000
    np.random.seed(42)
    strat_counts = s1_df["strat_key"].value_counts(normalize=True)
    sampled_indices = []

    for key, frac in strat_counts.items():
        sub_df = s1_df[s1_df["strat_key"] == key]
        n_sample = int(round(frac * benchmark_size))
        if n_sample > len(sub_df):
            n_sample = len(sub_df)
        sampled_indices.extend(sub_df.sample(n=n_sample, random_state=42).index.tolist())

    s1_sample_df = s1_df.loc[sampled_indices].copy()
    sampled_s1_ids = set(s1_sample_df["entity_id"])

    # Extract all true pairs for the sampled S1 entities
    true_pairs = []
    sampled_target_ids = set()
    for s1_id in sampled_s1_ids:
        for m_id in gt_map.get(s1_id, []):
            true_pairs.append((s1_id, m_id))
            sampled_target_ids.add(m_id)

    print(f"Sampled {len(s1_sample_df):,} S1 entities -> {len(true_pairs):,} Ground-Truth Pairs ({len(sampled_target_ids):,} unique targets).", flush=True)

    # Build entity lookup tables
    print("[2/4] Indexing Entity Attributes (Target Filtered)...", flush=True)
    s1_dict = s1_sample_df.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    
    s2_sub = s2_df[s2_df["entity_id"].isin(sampled_target_ids)].set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s3_sub = s3_df[s3_df["entity_id"].isin(sampled_target_ids)].set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    
    target_dict = {}
    target_dict.update(s2_sub)
    target_dict.update(s3_sub)

    # Construct paired DataFrame for vectorized normalization
    pair_rows = []
    for s1_id, t_id in true_pairs:
        s1_rec = s1_dict.get(s1_id, {})
        t_rec = target_dict.get(t_id, {})
        pair_rows.append({
            "s1_id": s1_id,
            "target_id": t_id,
            "s1_name": str(s1_rec.get("business_name", "") or ""),
            "s1_addr": str(s1_rec.get("business_address", "") or ""),
            "t_name": str(t_rec.get("business_name", "") or ""),
            "t_addr": str(t_rec.get("business_address", "") or ""),
        })

    pairs_df = pd.DataFrame(pair_rows)

    print("[3/4] Running Normalization N3, N4, N5 across 34,691 true pairs...", flush=True)
    
    # N3 Normalization
    t_n3_start = time.time()
    pairs_df["s1_name_N3"] = fast_normalize_N3(pairs_df["s1_name"])
    pairs_df["t_name_N3"] = fast_normalize_N3(pairs_df["t_name"])
    pairs_df["s1_addr_N3"] = fast_normalize_N3(pairs_df["s1_addr"])
    pairs_df["t_addr_N3"] = fast_normalize_N3(pairs_df["t_addr"])
    n3_time = time.time() - t_n3_start

    # N4 Normalization
    t_n4_start = time.time()
    pairs_df["s1_name_N4"] = fast_normalize_N4(pairs_df["s1_name"])
    pairs_df["t_name_N4"] = fast_normalize_N4(pairs_df["t_name"])
    pairs_df["s1_addr_N4"] = fast_normalize_N4(pairs_df["s1_addr"])
    pairs_df["t_addr_N4"] = fast_normalize_N4(pairs_df["t_addr"])
    n4_time = time.time() - t_n4_start

    # N5 Normalization
    t_n5_start = time.time()
    pairs_df["s1_name_N5"] = fast_normalize_N5(pairs_df["s1_name"])
    pairs_df["t_name_N5"] = fast_normalize_N5(pairs_df["t_name"])
    pairs_df["s1_addr_N5"] = fast_normalize_N5(pairs_df["s1_addr"])
    pairs_df["t_addr_N5"] = fast_normalize_N5(pairs_df["t_addr"])
    n5_time = time.time() - t_n5_start

    print("[4/4] Computing Pair Match Equalities & Character Similarities...", flush=True)

    # Equalities
    pairs_df["name_exact_N3"] = (pairs_df["s1_name_N3"] == pairs_df["t_name_N3"]) & (pairs_df["s1_name_N3"] != "")
    pairs_df["addr_exact_N3"] = (pairs_df["s1_addr_N3"] == pairs_df["t_addr_N3"]) & (pairs_df["s1_addr_N3"] != "")

    pairs_df["name_exact_N4"] = (pairs_df["s1_name_N4"] == pairs_df["t_name_N4"]) & (pairs_df["s1_name_N4"] != "")
    pairs_df["addr_exact_N4"] = (pairs_df["s1_addr_N4"] == pairs_df["t_addr_N4"]) & (pairs_df["s1_addr_N4"] != "")

    pairs_df["name_exact_N5"] = (pairs_df["s1_name_N5"] == pairs_df["t_name_N5"]) & (pairs_df["s1_name_N5"] != "")
    pairs_df["addr_exact_N5"] = (pairs_df["s1_addr_N5"] == pairs_df["t_addr_N5"]) & (pairs_df["s1_addr_N5"] != "")

    # Calculate similarities for a sample or vectorized where possible
    print("Computing character & token similarities for all true pairs...", flush=True)

    def calc_sims(row, level):
        sn = row[f"s1_name_{level}"]
        tn = row[f"t_name_{level}"]
        sa = row[f"s1_addr_{level}"]
        ta = row[f"t_addr_{level}"]

        return {
            f"name_char_{level}": char_sim(sn, tn),
            f"addr_char_{level}": char_sim(sa, ta),
            f"name_jaccard_{level}": token_jaccard(sn, tn),
            f"addr_jaccard_{level}": token_jaccard(sa, ta)
        }

    n3_sims = pairs_df.apply(lambda r: calc_sims(r, "N3"), axis=1, result_type="expand")
    n4_sims = pairs_df.apply(lambda r: calc_sims(r, "N4"), axis=1, result_type="expand")
    n5_sims = pairs_df.apply(lambda r: calc_sims(r, "N5"), axis=1, result_type="expand")

    pairs_df = pd.concat([pairs_df, n3_sims, n4_sims, n5_sims], axis=1)

    # N4 Impact Analysis (N4 vs N3)
    n4_addr_gained_exact = pairs_df[pairs_df["addr_exact_N4"] & ~pairs_df["addr_exact_N3"]]
    n4_addr_lost_exact = pairs_df[pairs_df["addr_exact_N3"] & ~pairs_df["addr_exact_N4"]]
    n4_addr_sim_increased = pairs_df[pairs_df["addr_char_N4"] > pairs_df["addr_char_N3"] + 0.05]
    n4_addr_sim_decreased = pairs_df[pairs_df["addr_char_N4"] < pairs_df["addr_char_N3"] - 0.05]

    # N5 Impact Analysis (N5 vs N4)
    n5_name_gained_exact = pairs_df[pairs_df["name_exact_N5"] & ~pairs_df["name_exact_N4"]]
    n5_addr_gained_exact = pairs_df[pairs_df["addr_exact_N5"] & ~pairs_df["addr_exact_N4"]]
    n5_name_sim_increased = pairs_df[pairs_df["name_char_N5"] > pairs_df["name_char_N4"] + 0.05]

    # Concrete Examples
    n4_examples = []
    for _, r in n4_addr_gained_exact.head(10).iterrows():
        n4_examples.append({
            "s1_id": r["s1_id"],
            "target_id": r["target_id"],
            "s1_addr_raw": r["s1_addr"],
            "t_addr_raw": r["t_addr"],
            "N3_s1_addr": r["s1_addr_N3"],
            "N3_t_addr": r["t_addr_N3"],
            "N4_normalized_addr": r["s1_addr_N4"]
        })

    n5_examples = []
    for _, r in n5_name_gained_exact.head(10).iterrows():
        n5_examples.append({
            "s1_id": r["s1_id"],
            "target_id": r["target_id"],
            "s1_name_raw": r["s1_name"],
            "t_name_raw": r["t_name"],
            "N4_s1_name": r["s1_name_N4"],
            "N4_t_name": r["t_name_N4"],
            "N5_normalized_name": r["s1_name_N5"]
        })

    total_pairs = len(pairs_df)

    diagnostic_results = {
        "benchmark_metadata": {
            "s1_sample_size": 10000,
            "total_ground_truth_pairs": total_pairs,
            "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "pair_equality_rates": {
            "N3": {
                "exact_name_equality_pct": float(pairs_df["name_exact_N3"].mean() * 100),
                "exact_address_equality_pct": float(pairs_df["addr_exact_N3"].mean() * 100),
                "name_char_sim_mean": float(pairs_df["name_char_N3"].mean()),
                "name_char_sim_median": float(pairs_df["name_char_N3"].median()),
                "addr_char_sim_mean": float(pairs_df["addr_char_N3"].mean()),
                "addr_char_sim_median": float(pairs_df["addr_char_N3"].median()),
            },
            "N4": {
                "exact_name_equality_pct": float(pairs_df["name_exact_N4"].mean() * 100),
                "exact_address_equality_pct": float(pairs_df["addr_exact_N4"].mean() * 100),
                "name_char_sim_mean": float(pairs_df["name_char_N4"].mean()),
                "name_char_sim_median": float(pairs_df["name_char_N4"].median()),
                "addr_char_sim_mean": float(pairs_df["addr_char_N4"].mean()),
                "addr_char_sim_median": float(pairs_df["addr_char_N4"].median()),
                "runtime_seconds": float(n4_time)
            },
            "N5": {
                "exact_name_equality_pct": float(pairs_df["name_exact_N5"].mean() * 100),
                "exact_address_equality_pct": float(pairs_df["addr_exact_N5"].mean() * 100),
                "name_char_sim_mean": float(pairs_df["name_char_N5"].mean()),
                "name_char_sim_median": float(pairs_df["name_char_N5"].median()),
                "addr_char_sim_mean": float(pairs_df["addr_char_N5"].mean()),
                "addr_char_sim_median": float(pairs_df["addr_char_N5"].median()),
                "runtime_seconds": float(n5_time)
            }
        },
        "n4_impact_analysis": {
            "pairs_gained_exact_address_equality": len(n4_addr_gained_exact),
            "pairs_lost_exact_address_equality": len(n4_addr_lost_exact),
            "pairs_address_similarity_increased": len(n4_addr_sim_increased),
            "pairs_address_similarity_decreased": len(n4_addr_sim_decreased),
            "concrete_examples": n4_examples
        },
        "n5_impact_analysis": {
            "pairs_gained_exact_name_equality": len(n5_name_gained_exact),
            "pairs_gained_exact_address_equality": len(n5_addr_gained_exact),
            "pairs_name_similarity_increased": len(n5_name_sim_increased),
            "concrete_examples": n5_examples
        }
    }

    # Save to scratch/n4_n5_pair_diagnostic.json
    with open("scratch/n4_n5_pair_diagnostic.json", "w") as f:
        json.dump(diagnostic_results, f, indent=2)

    print("\nSaved Diagnostic JSON to scratch/n4_n5_pair_diagnostic.json", flush=True)

    # Print summary report
    print("\n" + "="*70, flush=True)
    print("DIAGNOSTIC SUMMARY REPORT", flush=True)
    print("="*70, flush=True)
    print(f"N3 -> Exact Name: {diagnostic_results['pair_equality_rates']['N3']['exact_name_equality_pct']:.2f}%, Exact Addr: {diagnostic_results['pair_equality_rates']['N3']['exact_address_equality_pct']:.2f}%", flush=True)
    print(f"N4 -> Exact Name: {diagnostic_results['pair_equality_rates']['N4']['exact_name_equality_pct']:.2f}%, Exact Addr: {diagnostic_results['pair_equality_rates']['N4']['exact_address_equality_pct']:.2f}%", flush=True)
    print(f"N5 -> Exact Name: {diagnostic_results['pair_equality_rates']['N5']['exact_name_equality_pct']:.2f}%, Exact Addr: {diagnostic_results['pair_equality_rates']['N5']['exact_address_equality_pct']:.2f}%", flush=True)
    print(f"N4 Gained Exact Addresses: {len(n4_addr_gained_exact)} pairs", flush=True)
    print(f"N5 Gained Exact Names: {len(n5_name_gained_exact)} pairs", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
