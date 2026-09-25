# Milestone 2 Audit & Stress Test Report

**Project**: Amazon ML Challenge 2026 — Business Entity Resolution Pipeline  
**Date**: 2026-09-25  
**Audit Scope**: Baseline Codebase Verification, Stress Testing, and Invariant Audit

---

## 1. Current Architecture [PASS]

The pipeline follows a clean, unidirectional 7-stage design:

```text
Data Loading ──► Profiling ──► Normalization ──► Candidate Blocking ──► Feature Engineering ──► Matcher ──► Output & Validation
```

* **Modularity**: All stages are strictly decoupled into single-responsibility Python modules in `src/`.
* **State Management**: Data flows as immutable DataFrames; no global mutable state.

---

## 2. Normalization Audit [PASS / WARNING]

### Implemented Behavior
* Lowercasing, Unicode NFKD normalization (removing accents e.g., `José` -> `jose`), character stripping `[^\w\s]`, and whitespace normalization.
* Non-destructive legal suffix extraction into `name_core_tokens` and `name_suffix_tokens`.

### Audit Findings
* `[PASS]` Handles lowercasing, accents, whitespace, special characters, numbers-only, short strings, and legal suffixes without throwing exceptions.
* `[WARNING]` **Acronyms & Concatenated Names**: Strings like `"ABC-TECH"` become `"abc tech"`. If another source keeps `"abctech"`, token-based matching will fail.
* `[WARNING]` **Abbreviations**: `"12 MG Road"` vs `"12 Mahatma Gandhi Road"` — normalization does not expand common street or business abbreviations (`mg` -> `mahatma gandhi`, `rd` -> `road`, `st` -> `street`).

---

## 3. Blocking Audit [PASS / WARNING]

### Implemented Strategies & Analysis

| Strategy | Blocking Key | Failure Cases | False Candidate Risks | Candidate Recall Impact |
|---|---|---|---|---|
| **Block A: Exact Name** | `(country, name_norm)` | Spelling typos, word-order changes, legal suffix variations. | Multiple businesses sharing generic names. | Missed if 1 character differs. |
| **Block B: Name Token** | `(country, token)` (`len>=3`) | Short 2-letter tokens (`HP`, `3M`, `GE`, `AI`) filtered out. | Massive candidate explosions on generic tokens (`global`, `trading`, `solutions`). | Missed for ultra-short names. |
| **Block C: Address Token** | `(country, token)` (`digit` or `len>=4`) | Address format differences (`12 MG Rd` vs `12 Mahatma Gandhi Road`). | Candidate explosions on common house numbers (`1`, `10`, `12`) or city names. | Missed if address missing or abbreviated. |
| **Block D: Char Signature** | `(country, sorted(chars[:3]))` | Prefix changes (`The ABC Corp` vs `ABC Corp`), transliteration. | High candidate generation for names sharing common initial letter sets. | Missed if initial 3 characters change. |

### Audit Findings
* `[WARNING]` **Strict Country Constraint**: Indexing key is `(country, token)`. If a noisy source has a misspelled country (`U.S.A.` vs `USA`, or missing country), **ALL candidates will be missed**.
* `[WARNING]` **Stopword & Token Explosion**: High-frequency tokens (`solutions`, `services`, `group`) create large candidate sets. TF-IDF thresholding or stopword suppression will be required for scalability on the full dataset.

---

## 4. Feature Audit [PASS]

### Calculated Features
* **Name**: `name_exact`, `name_jaccard`, `name_char_similarity`, `name_edit_similarity`, `name_token_overlap`, `name_len_diff`, `name_len_ratio`.
* **Address**: `address_exact`, `address_jaccard`, `address_char_similarity`, `address_edit_similarity`, `address_token_overlap`, `address_len_diff`, `numeric_token_overlap`.
* **Country**: `country_match`.

### Audit Findings
* `[PASS]` `numeric_token_overlap` correctly captures matching street/building numbers between addresses.
* `[PASS]` `RapidFuzz` Levenshtein ratios provide normalized `[0.0, 1.0]` string similarities.

---

## 5. Matcher Audit [PASS / WARNING]

### Formula
$$\text{Score} = 0.50 \cdot \text{Name}_{\text{composite}} + 0.35 \cdot \text{Address}_{\text{composite}} + 0.15 \cdot \text{Country}_{\text{match}}$$
where:
$$\text{Name}_{\text{composite}} = 0.50 \cdot \text{Name}_{\text{char\_sim}} + 0.30 \cdot \text{Name}_{\text{jaccard}} + 0.20 \cdot \text{Name}_{\text{exact}}$$
$$\text{Address}_{\text{composite}} = 0.50 \cdot \text{Address}_{\text{char\_sim}} + 0.30 \cdot \text{Address}_{\text{jaccard}} + 0.20 \cdot \text{Address}_{\text{exact}}$$

### Controlled Scoring Stress Test Results

| Case | Name Sim | Address Sim | Country Match | Final Score | Match Decision ($\text{Threshold}=0.70$) |
|---|---|---|---|---|---|
| **1. Very Strong Match** | 0.78 | 1.00 | 1 | **0.7159** | `TRUE` |
| **2. Strong Name / Weak Address** | 0.78 | 0.30 | 1 | **0.4191** | `FALSE` |
| **3. Weak Name / Strong Address** | 0.30 | 1.00 | 1 | **0.5741** | `FALSE` |
| **4. Same Name / Different Country** | 1.00 | 0.70 | 0 | **0.6860** | `FALSE` |
| **5. Completely Unrelated** | 0.29 | 0.34 | 0 | **0.1312** | `FALSE` |

### Audit Findings
* `[PASS]` Scoring produces intuitive scalar rankings.
* `[WARNING]` **Heuristic Weights**: Fixed manual weights ($0.50, 0.35, 0.15$) and threshold ($0.70$) are baselines only. On noisy data where addresses are missing, Case 2 ("Strong Name / Weak Address") gets score $0.4191$ and is rejected, which may hurt recall. Weight optimization must wait for official training data.

---

## 6. F0.5 Evaluation Audit [PASS]

### Formula Verification
$$F_{0.5} = (1 + 0.5^2) \frac{\text{Precision} \cdot \text{Recall}}{(0.5^2 \cdot \text{Precision}) + \text{Recall}} = 1.25 \frac{\text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

* `[PASS]` Verified via unit test `test_f05_formula_exactness()` (TP=4, FP=1, FN=1 -> Precision=0.8, Recall=0.8, $F_{0.5}=0.8000$).
* `[PASS]` Correctly accounts for singletons, multi-matches, and missing ground-truth matches.

---

## 7. Output Contract Audit [PASS]

### Checks Performed
1. `matching_results.tsv`:
   * Exactly 2 columns (`entity_id`, `matched_entity_ids`).
   * Every Source 1 entity appears exactly once.
   * Multi-matches comma-separated without spaces (e.g., `S2_201,S3_301`).
   * No-match entities output empty string `""`.
2. `candidate_pairs.tsv`:
   * Columns: `source1_entity_id`, `candidate_entity_id`, `candidate_source`, `blocking_rules`.
   * **Hard Invariant**: `assert all(final_matches ⊆ candidate_pairs)` holds.

---

## 8. Determinism Test [PASS]

* Executed pipeline multiple times sequentially.
* `pd.testing.assert_frame_equal` confirmed **100% bit-for-bit identity** across candidate generation, feature values, scores, and exported TSV files.

---

## 9. Runtime Baseline [PASS]

Executed on mock dataset (29 test cases):

| Stage | Execution Time |
|---|---|
| **Normalization** | ~6.0 ms |
| **Blocking** | ~1.2 ms |
| **Feature Extraction** | ~4.0 ms |
| **Matching Engine** | ~2.3 ms |
| **Post-processing** | ~0.5 ms |
| **Total Pipeline** | **~14.0 ms** |

---

## 10. Bugs Found & Fixed

1. **Bug #1 (Fixed)**: `validate_pipeline_outputs` raised `KeyError: 'entity_id'` when an empty DataFrame (e.g. `s3_df`) was passed without explicit columns.
   * *Fix*: Added column existence check `s2_ids = set(s2_df["entity_id"]) if "entity_id" in s2_df.columns and not s2_df.empty else set()`.
2. **Bug #2 (Fixed)**: `normalize_dataframe` raised `KeyError: 'business_name'` when provided an empty DataFrame.
   * *Fix*: Initialized missing required columns `['business_name', 'business_address', 'country']` to empty string if missing in `normalize_dataframe`.

---

## 11. Known Limitations

1. **Country Soft-Matching**: Strictly filtering candidates on exact normalized country means country transliterations or typos will drop true candidate pairs.
2. **Short Entity Names**: 2-letter tokens (e.g., `3M`, `HP`) are ignored by token blocking (`min_token_len=3`).
3. **High-Frequency Stopwords**: Generic tokens (`trading`, `global`) will create large candidate pairs on full datasets unless suppressed.

---

## 12. What Should Wait Until Official Dataset

1. Supervised Machine Learning model training (LightGBM/XGBoost/CatBoost).
2. TF-IDF indexer / MinHash LSH tuning for scaling candidate generation.
3. Optimal threshold & weight selection via grid search on `train_ground_truth.tsv`.
4. Domain-specific legal suffix and abbreviation replacement maps derived from real data distribution.
