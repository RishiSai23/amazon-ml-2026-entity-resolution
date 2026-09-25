# Amazon ML Challenge 2026 — Business Entity Resolution Pipeline

A clean, modular, experiment-friendly entity-resolution pipeline to match business entities across deduplicated reference sources (Source 1) and noisy sources (Source 2 and Source 3).

## Project Overview

* **Goal**: Match Source 1 entities to zero, one, or multiple entities in Source 2 and Source 3.
* **Evaluation Metric**: **F0.5** (emphasizes Precision over Recall).
* **Strict Restrictions**: No external datasets, no commercial entity resolution APIs, no geocoding APIs, no internet searches.

## Project Structure

```text
amazon-ml/
├── dataset/
│   ├── train/            # Challenge training files (or mock unit-test data)
│   └── test/             # Challenge test files
├── src/
│   ├── __init__.py
│   ├── config.py         # Pipeline configuration dataclass & settings
│   ├── io_utils.py       # Data loading, schema validation, and profiling
│   ├── normalize.py      # Non-destructive text normalization & tokenization
│   ├── blocking.py       # Multi-strategy candidate generation engine
│   ├── features.py       # Pairwise similarity feature extraction
│   ├── matching.py       # Rule-based / weighted matching decision engine
│   ├── postprocess.py    # Formatting & invariant validation
│   ├── generate_output.py# File export writer for target TSVs
│   └── main.py           # CLI driver script
├── experiments/          # Tracking logs for experimental iterations
├── output/               # Produced matching_results.tsv and candidate_pairs.tsv
├── tests/                # Automated pytest suite
├── requirements.txt      # Python dependencies
└── README.md
```

## Quick Start

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run End-to-End Pipeline (Mock Dataset)
```bash
python src/main.py --mode mock
```

### Run Pipeline on Actual Train/Test Data
```bash
# Run on train dataset with ground truth evaluation
python src/main.py --mode train

# Run on test dataset and produce final output/ files
python src/main.py --mode test
```

### Run Tests
```bash
pytest tests/
```
