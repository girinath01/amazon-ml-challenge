# Business Entity Resolution - End-to-End Pipeline

This directory contains the source code and instructions to reproduce the end-to-end pipeline (Data Preparation → Blocking → Matching → Output Generation).

## Setup & Installation

```bash
pip install -r requirements.txt
```

## How to Reproduce End-to-End

1. **Blocking / Candidate Generation:**
   Runs candidate generation and outputs `output/candidate_pairs.tsv`.

2. **Matching & Model Inference:**
   Runs matching model on generated candidates and outputs `output/matching_results.tsv`.
