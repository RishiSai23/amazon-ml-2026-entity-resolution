# Experiment Tracking Log

| Experiment ID | Date | Blocking Strategy | Features | Matcher Threshold | Candidate Count | Candidate Recall | Precision | Recall | F0.5 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-09-25 | Exact Name + Prefix + Address + CharSig | Weighted Baseline (Name, Addr, Country) | 0.85 | - | - | - | - | - | Initial mock baseline |

## Guidelines for Experiments

1. **Precision Focus (F0.5)**: False positives cost significantly more than false negatives. Always prioritize high precision.
2. **Submission Limits**: Maximum 5 submissions per day. Evaluate locally on `train_ground_truth.tsv` before submitting.
3. **Candidate Recall vs Matcher Precision**: Record candidate recall separately from matcher precision/recall to isolate blocking issues from decision layer issues.
