# Milestone 3 — Official Dataset Forensics

## 1. Dataset Overview
The official Amazon ML Challenge Business Entity Resolution dataset has arrived. The dataset contains **12,527,040 training records** across Source 1, Source 2, and Source 3, along with **11,702,133 test records** and **7,638,365 true pair matches** documented in the ground truth file.

The primary objective of this forensic analysis is to profile the raw properties of the official dataset without altering existing pipeline code, tuning thresholds, or training predictive models.

---

## 2. File Validation
All 7 expected TSV files are present, well-formed, and strictly follow tabular TSV structure with `\t` delimiters.

| File Path | Exists | Size (MB) | Format | Header / Columns Verified |
| :--- | :---: | ---: | :---: | :---: |
| `dataset/train/train_source1.tsv` | Yes | 200.34 MB | TSV | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/train/train_source2.tsv` | Yes | 466.63 MB | TSV | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/train/train_source3.tsv` | Yes | 480.37 MB | TSV | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/train/train_ground_truth.tsv` | Yes | 121.13 MB | TSV | `source1_entity_id`, `matched_entity_ids` |
| `dataset/test/test_source1.tsv` | Yes | 166.91 MB | TSV | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/test/test_source2.tsv` | Yes | 485.86 MB | TSV | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/test/test_source3.tsv` | Yes | 482.56 MB | TSV | `entity_id`, `business_name`, `business_address`, `country` |

---

## 3. Dataset Dimensions

| Source | Rows | Unique IDs | Duplicate IDs | File Size (MB) |
| :--- | ---: | ---: | ---: | ---: |
| **Train S1** | 2,206,821 | 2,206,821 | 0 | 200.34 MB |
| **Train S2** | 5,034,616 | 5,034,616 | 0 | 466.63 MB |
| **Train S3** | 5,285,603 | 5,285,603 | 0 | 480.37 MB |
| **Train Ground Truth** | 2,206,821 | 2,206,821 | 0 | 121.13 MB |
| **Test S1** | 1,732,544 | 1,732,544 | 0 | 166.91 MB |
| **Test S2** | 4,887,273 | 4,887,273 | 0 | 485.86 MB |
| **Test S3** | 5,082,316 | 5,082,316 | 0 | 482.56 MB |

> [!NOTE]
> Every entity source file has 100% unique primary keys (`entity_id`). There are zero duplicate entity IDs within any single source file.

---

## 4. Missing Values

### Source Column Missingness Breakdown

| File | Null `entity_id` | Null `business_name` | Null `business_address` | Null `country` | `business_address` Missing % |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Train S1** | 0 | 0 | 0 | 0 | 0.00% |
| **Train S2** | 0 | 2 | 168,967 | 0 | 3.36% |
| **Train S3** | 0 | 13 | 175,916 | 0 | 3.33% |
| **Test S1** | 0 | 0 | 0 | 0 | 0.00% |
| **Test S2** | 0 | 46 | 129,408 | 0 | 2.65% |
| **Test S3** | 0 | 59 | 136,098 | 0 | 2.68% |

---

## 5. Country Distribution

> [!CRITICAL]
> **Major Domain Shift Alert**: The training dataset contains only **2 countries** (`US` and `India`), whereas the test dataset introduces a **3rd country (`France`)**, accounting for **14.4% to 15.0%** of all test records!

### Country Record Counts & Percentages

| Source | US Count (%) | India Count (%) | France Count (%) | Total Rows |
| :--- | ---: | ---: | ---: | ---: |
| **Train S1** | 1,323,633 (60.0%) | 883,188 (40.0%) | 0 (0.0%) | 2,206,821 |
| **Train S2** | 3,016,817 (59.9%) | 2,017,799 (40.1%) | 0 (0.0%) | 5,034,616 |
| **Train S3** | 3,170,056 (60.0%) | 2,115,547 (40.0%) | 0 (0.0%) | 5,285,603 |
| **Test S1** | 663,106 (38.3%) | 809,986 (46.8%) | 259,452 (15.0%) | 1,732,544 |
| **Test S2** | 1,871,330 (38.3%) | 2,312,565 (47.3%) | 703,378 (14.4%) | 4,887,273 |
| **Test S3** | 1,945,701 (38.3%) | 2,405,000 (47.3%) | 731,615 (14.4%) | 5,082,316 |

- Exact country strings in Train: `"US"`, `"India"`
- Exact country strings in Test: `"US"`, `"India"`, `"France"`
- Casing differences: None found (all clean ISO/Standard casing).
- Missing country values: 0 missing values across all files.

---

## 6. Business Name Analysis

### Statistical Summary of Raw Names

| Source | Min Len | Max Len | Median Len | Avg Len | Punctuation % | Digits % | Non-ASCII % | Legal Suffix % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Train S1** | 3 | 105 | 24.0 | 24.03 | 20.92% | 1.61% | 0.00% | 64.34% |
| **Train S2** | 0 | 104 | 25.0 | 25.10 | 42.88% | 5.08% | 15.19% | 52.99% |
| **Train S3** | 0 | 123 | 25.0 | 25.20 | 40.10% | 5.12% | 11.48% | 55.31% |
| **Test S1** | 3 | 92 | 24.0 | 23.84 | 18.32% | 1.17% | 2.35% | 59.37% |
| **Test S2** | 0 | 102 | 25.0 | 25.70 | 40.94% | 3.89% | 18.99% | 49.32% |
| **Test S3** | 0 | 103 | 25.0 | 25.66 | 37.39% | 3.98% | 14.51% | 52.49% |

### Non-ASCII Character Scripts & Accents
- **Train S1**: 100% pure ASCII.
- **Train S2 & S3**: Contain Devanagari (`राम मार्केटिंग`), Gujarati (`શક્તિ અર્બન`), Tamil (`குளோபல் பிசினஸ்`), Malayalam (`സെന്റർ`), and accented Latin characters (`Établissements`, `SARL`).
- **Test Set**: High frequency of French accents (`é`, `è`, `à`, `ç`, `ê`) due to the inclusion of French business entities.

### Representative Noisy Business Names (30 Examples)
1. `50/53 Mobility`
2. `E/I World`
3. `H/H Artificial Inc`
4. `#4 Presidio`
5. `D/Z Bancshares LLC`
6. `17/66 Pro`
7. `# 3 NM Uranium`
8. `#Ninaor Rmg-Westminster`
9. `राम मार्केटिंग प्राइवेट लिमिटेड`
10. `आदित्य प्रॉपर्टीज एलएलपी`
11. `SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com`
12. `सन कंस्ट्रक्शंस प्राइवेट लिमिटेड`
13. `रियल मॉडर्न फूड लिमिटेड`
14. `குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்`
15. `Animal Welfare Nétwork`
16. `Private Ambernath Sólar Limited`
17. `શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ`
18. `અલ્ફા અલ ટેક્નોલોજીજ પ્રાઇવેટ લિમિટેડ`
19. `LLC Moncada Léarning Center`
20. `Béque`
21. `#centraleducation`
22. `அரிஹந்த் Foundation Private Limited`
23. `Cardiology Heartland Cára Associates #98825`
24. `സിൽവർ കൺസൾട്ടൻസി പ്രൈവറ്റ് ലിമിറ്റഡ്`
25. `ब्लू टेक्नोलॉजीज`
26. `గుజరాత్ Logistics లిమిటెడ్`
27. `M/s Jd Pvt Ltd Services`
28. `Secure Secure Cárolina`
29. `<< Team Ecole`
30. `Établissements Demployeurs SARL`

---

## 7. Address Analysis

### Statistical Summary of Raw Addresses

| Source | Missing % | Min Len | Max Len | Median Len | Avg Len | Contains Numbers % | Avg Digits | Non-ASCII % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Train S1** | 0.00% | 11 | 256 | 41.0 | 52.07 | 96.51% | 4.04 | 0.03% |
| **Train S2** | 3.36% | 0 | 249 | 37.0 | 46.23 | 90.65% | 3.88 | 9.50% |
| **Train S3** | 3.33% | 0 | 240 | 42.0 | 46.71 | 90.80% | 3.90 | 9.02% |
| **Test S1** | 0.00% | 11 | 268 | 50.0 | 57.21 | 95.85% | 3.69 | 4.26% |
| **Test S2** | 2.65% | 0 | 269 | 43.0 | 50.41 | 92.67% | 3.71 | 14.75% |
| **Test S3** | 2.68% | 0 | 267 | 43.0 | 48.74 | 92.49% | 3.65 | 14.35% |

### Common Address Patterns & Abbreviations
- **US Streets**: `St`, `Street`, `Rd`, `Road`, `Ave`, `Avenue`, `Blvd`, `Dr`, `Drive`, `Pkwy`, `Suite`, `Ste`, `Apt`, `Unit`.
- **India Streets**: `Marg`, `Nagar`, `Chowk`, `Gali`, `Plot No`, `Shop No`, `Floor`, `Bhavan`, `Industrial Area`, `MIDC`.
- **France Streets**: `Rue`, `Avenue`, `Boulevard`, `Allée`, `Place`, `Route`, `Impasse`, `BP`, `CEDEX`.

---

## 8. Ground Truth Structure

`train_ground_truth.tsv` consists of **2,206,821 rows** matching exact 1:1 with `train_source1.tsv`.
- Column 1: `source1_entity_id`
- Column 2: `matched_entity_ids` (comma-separated list of entity IDs from Source 2 and Source 3, or `NULL` if no match exists).

Total true pair matches across all S1 entities: **7,638,365 pairs**.

---

## 9. Match Cardinality

Distribution of true match count per Source 1 entity:

| Match Count | S1 Entity Count | Percentage |
| :--- | ---: | ---: |
| **0 matches** (No match) | 123,247 | 5.58% |
| **1 match** (Single match) | 119,157 | 5.40% |
| **2 matches** | 375,212 | 17.00% |
| **3 matches** | 530,841 | 24.05% |
| **4+ matches** | 1,058,364 | 47.96% |
| **Total** | **2,206,821** | **100.00%** |

### True Match Target Source Breakdown
- Matches in **Both Source 2 and Source 3**: 1,776,047 S1 entities (**85.24%** of matched entities)
- Matches in **Source 3 only**: 164,498 S1 entities (**7.89%**)
- Matches in **Source 2 only**: 143,029 S1 entities (**6.86%**)

---

## 10. True Match Similarity Analysis

Sample analysis of **30,000 true match pairs** from Ground Truth:

| Metric | Mean | Median | P25 | P75 | P90 | Exact Equality % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Name Character Similarity** | 0.786 | 0.865 | 0.722 | 0.947 | 1.000 | **10.71%** |
| **Name Token Similarity** | 0.562 | 0.600 | 0.333 | 0.750 | 1.000 | — |
| **Address Character Similarity** | 0.752 | 0.826 | 0.645 | 0.919 | 1.000 | **7.43%** |
| **Address Token Similarity** | 0.520 | 0.500 | 0.333 | 0.714 | 1.000 | — |
| **Country Match** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.00%** |

---

## 11. Negative Pair Analysis

Sample analysis of **30,000 non-matching (random negative) pairs**:

| Metric | Mean | Median | P25 | P75 | P90 | Exact Equality % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Name Character Similarity** | 0.307 | 0.311 | 0.256 | 0.364 | 0.419 | **0.00%** |
| **Name Token Similarity** | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| **Address Character Similarity** | 0.322 | 0.326 | 0.288 | 0.366 | 0.450 | **0.00%** |
| **Address Token Similarity** | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| **Country Match** | 52.18% | 100.0% | 0.0% | 100.0% | 100.0% | **52.18%** |

> [!KEY FINDING]
> True matches have a median name character similarity of **0.865** vs **0.311** for negative pairs. Token overlap on raw un-normalized text drops sharply (median 0.600 for true matches), demonstrating why normalization and token alignment are critical.

---

## 12. Duplicate Analysis

| Source | Total Rows | Duplicate Name Rows (%) | Duplicate Address Rows (%) | Duplicate (Name + Addr) Rows (%) | Duplicate Full Records (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Train S1** | 2,206,821 | 845,385 (38.31%) | 116,304 (5.27%) | 0 (0.00%) | 0 (0.00%) |
| **Train S2** | 5,034,616 | 872,386 (17.33%) | 1,118,827 (22.22%) | 50,969 (1.01%) | 50,933 (1.01%) |
| **Train S3** | 5,285,603 | 892,270 (16.88%) | 1,035,729 (19.60%) | 37,283 (0.71%) | 37,241 (0.70%) |
| **Test S1** | 1,732,544 | 623,632 (36.00%) | 86,457 (4.99%) | 0 (0.00%) | 0 (0.00%) |
| **Test S2** | 4,887,273 | 799,335 (16.36%) | 1,107,188 (22.65%) | 44,552 (0.91%) | 44,550 (0.91%) |
| **Test S3** | 5,082,316 | 798,543 (15.71%) | 1,030,637 (20.28%) | 32,178 (0.63%) | 32,154 (0.63%) |

---

## 13. Blocking Analysis

Evaluated across all **7,638,365 true GT pairs**:

| Blocking Strategy | Candidate Pairs Captured | Total True GT Pairs | Candidate Recall |
| :--- | ---: | ---: | ---: |
| **Block A — Exact Name** | 1,668,793 | 7,638,365 | **21.85%** |
| **Block B — Name Token** | 6,439,420 | 7,638,365 | **84.30%** |
| **Block C — Address Token** | 7,295,990 | 7,638,365 | **95.52%** |
| **Block D — Character Signature** | 5,909,192 | 7,638,365 | **77.36%** |
| **Block E — Union** | **7,636,449** | **7,638,365** | **99.97%** |

> [!IMPORTANT]
> The **Union blocking strategy** captures **99.97%** of all 7.64 million true match pairs in the dataset. Out of 7,638,365 true matches, only 1,916 pairs are missed by the current union blocking scheme.

---

## 14. Blocking Failure Cases

### False Negatives (Missed True Matches)
1. **Script Mismatches**: S1 entity in English script (e.g. `Ram Marketing`) vs S2 true match in Devanagari script (`राम मार्केटिंग`). Blocking strategies using raw ASCII tokens miss 100% of non-transliterated foreign script matches.
2. **Abbreviation Discrepancies**: S1 entity `St. Jude Medical Center` vs S2 true match `Saint Jude Med Ctr`. Token-based blocking misses blocks when tokens are shortened or expanded without canonical normalization.
3. **Missing Address Field**: 3.36% of S2 and 3.33% of S3 entities have `NULL` addresses. Address-token blocking misses 100% of true matches for these entities.

### False Positives (Candidate Explosion)
1. **Generic Legal Tokens**: Words like `Pvt`, `Ltd`, `Inc`, `LLC`, `Company`, `Services` produce millions of candidate pair collisions if not excluded as stop words during token blocking.
2. **Common Address Keywords**: Words like `Road`, `Street`, `Avenue`, `Building`, `Floor`, `Suite`, `Opposite` create massive candidate explosion across unrelated businesses in the same city.

---

## 15. Train/Test Distribution Comparison

1. **Row Count Ratio**: Test S1 is ~78.5% the size of Train S1. Test S2 and S3 are ~96-97% the size of Train S2 and S3.
2. **New Country Domain**: Test introduces `France` (14.4% - 15.0% of test records), while Train has 0 French entities.
3. **Missing Address Rate**: Missing address rate in Test S2/S3 is ~2.65% to 2.68%, slightly lower than Train S2/S3 (~3.33% to 3.36%).
4. **Unicode / Non-ASCII Frequency**: Non-ASCII character frequency increases from 15.19% in Train S2 to 18.99% in Test S2 due to French accent characters (`é`, `è`, `à`, `ç`).
5. **Name / Address Lengths**: Median and average character lengths for names (24-25 chars) and addresses (41-50 chars) are highly consistent across train and test sets.

---

## 16. Baseline Performance

> [!NOTE]
> Exhaustive pair-level calculation across the full 12.5M entity dataset was stopped per explicit user directive to avoid computational inefficiency. As documented in [`experiments/milestone_3_decision.md`](file:///d:/Projects/AWS_ML_CHALLENGE/amazon-ml/experiments/milestone_3_decision.md), evaluation will transition to a statistically representative stratified sample (e.g., 50,000 S1 entities across US and India).
>
> On the exact candidate recall test across all **7,638,365 true ground truth pairs**, the **Union blocking strategy** achieved **99.97% candidate recall** (7,636,449 / 7,638,365 true matches captured).

---

## 17. Runtime

| Stage | Duration / Estimate | Method |
| :--- | ---: | :--- |
| **Data Loading (12.5M rows)** | 39.90s | Polars multi-threaded TSV reader |
| **Text Normalization** | 173.64s | vectorized string cleaning |
| **Inverted Index Construction** | 435.48s | Python Dict of Sets |
| **Full Exhaustive Evaluation** | Terminated | Replaced by Stratified Sampling |

---

## 18. Key Findings

1. **Massive Multi-Match Prevalence (89.0%)**: Over 89.0% of Source 1 entities have multiple valid matching records in S2 and S3. A matcher that stops at the top-1 prediction will fail severely on Recall.
2. **Dual-Source Coverage (85.2%)**: 85.24% of S1 entities match candidates in *both* Source 2 and Source 3 simultaneously.
3. **Unseen Country Shift (`France`)**: Test set introduces 14.4% French entities. Any country-specific hardcoding or rules will fail if not generalized to French language patterns.
4. **Non-ASCII Script Multi-Lingual Corruptions**: Indic scripts and French accented Latin text account for up to 19% of S2/S3 entity records. Transliteration and Unicode normalization are mandatory.
5. **Exact Match Failure**: Only 10.71% of true matches have identical business names, and only 7.43% have identical addresses. Fuzzy matching and smart feature alignment are indispensable.

---

## 19. Recommended Next Steps

Based strictly on dataset forensics:
1. **A. Better Normalization**: Implement Unicode/NFD normalization, accent stripping, lowercasing, script transliteration, and legal suffix removal.
2. **B. Multi-Strategy Blocking Optimization**: Upgrade token indexing with stop-word filtering for generic words (`Ltd`, `Inc`, `Street`, `Road`) to prevent candidate explosion while maximizing recall.
3. **C. TF-IDF & Character N-gram Indexing**: Introduce TF-IDF / N-gram cosine similarity for robust fuzzy candidate generation.
