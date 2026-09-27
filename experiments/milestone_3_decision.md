# Milestone 3 Decision — Stratified Evaluation Methodology

## 1. Context & Rationale

During Milestone 3, we completed dataset profiling and verified candidate recall for all five blocking strategies across all **7,638,365 true ground truth pairs**. The union blocking strategy achieved an outstanding **99.97% candidate recall** (capturing 7,636,449 out of 7,638,365 true matches).

However, full exhaustive baseline evaluation over the entire **12.5 million entity dataset** (2.2M S1, 5.0M S2, 5.3M S3) requires generating, indexing, scoring, and sorting tens to hundreds of millions of candidate pairs. In Python, this exhaustive computation takes tens of minutes to hours per pipeline run, creating an severe bottleneck for rapid iteration.

Per explicit directive, **task-473 was terminated**, and we are adopting a **statistically representative stratified sampling evaluation methodology** for all subsequent pipeline iterations and model benchmarking.

---

## 2. Completed Blocking Evaluation Highlights

Even on the full dataset, the blocking evaluation yielded definitive empirical metrics across all 7.64 million true pairs:

| Blocking Strategy | Captured True Pairs | Total GT Pairs | Candidate Recall | Assessment |
| :--- | ---: | ---: | ---: | :--- |
| **Block A — Exact Name** | 1,668,793 | 7,638,365 | **21.85%** | Poor; fails on minor spelling variants |
| **Block B — Name Token** | 6,439,420 | 7,638,365 | **84.30%** | Strong name signal |
| **Block C — Address Token** | 7,295,990 | 7,638,365 | **95.52%** | Highest single-field recall |
| **Block D — Character Signature** | 5,909,192 | 7,638,365 | **77.36%** | Good backup for typos |
| **Block E — Union** | **7,636,449** | **7,638,365** | **99.97%** | **Near-perfect candidate coverage** |

---

## 3. Proposed Sampled & Stratified Evaluation Methodology

To enable fast, reproducible, and mathematically sound model evaluation during Milestones 4+, we establish the following benchmarking protocol:

### A. Stratified Sampling Architecture
- **Sample Size**: $N = 50,000$ Source 1 entities (approx. 2.27% of `train_source1.tsv`).
- **Country Stratification**:
  - `US`: 30,000 entities (60.0%)
  - `India`: 20,000 entities (40.0%)
- **Match Cardinality Stratification**:
  - 0 matches: 5.58% (~2,790 entities)
  - 1 match: 5.40% (~2,700 entities)
  - 2 matches: 17.00% (~8,500 entities)
  - 3 matches: 24.05% (~12,025 entities)
  - 4+ matches: 47.96% (~23,985 entities)
- **Sample True Pairs**: ~173,000 ground truth pair matches.

### B. Benchmark Metrics
For every future pipeline change, evaluation will measure:
1. **Candidate Recall**: Percentage of ground truth matches in the sample captured by blocking.
2. **Candidate Count Distribution**: Mean, median, 95th percentile, and max candidate pairs per S1 query.
3. **Precision**: $\frac{TP}{TP + FP}$
4. **Recall**: $\frac{TP}{TP + FN}$
5. **F0.5 Score**: $\frac{1.25 \cdot Precision \cdot Recall}{0.25 \cdot Precision + Recall}$ (Primary competition metric).
6. **Execution Wall-Clock Time**: Seconds taken for indexing, candidate generation, feature extraction, and matching on the 50k benchmark sample.

---

## 4. Summary of Preserved Forensic Insights

1. **Multi-Match Dominance**: 89.02% of S1 entities have multiple true matches across S2 and S3; 85.24% match entities in *both* S2 and S3.
2. **Domain Shift Alert**: Test data introduces **14.4% French entities**, absent from training data. All normalization rules must be script-agnostic and handle French diacritics (`é`, `è`, `à`, `ç`).
3. **Non-ASCII & Indic Scripts**: S2 and S3 contain Indic scripts (Devanagari, Gujarati, Tamil, Malayalam) and French accents in up to 19% of records.
4. **Fuzzy Match Requirement**: Exact string matches occur in only 10.71% of names and 7.43% of addresses. Fuzzy feature engineering is essential.
