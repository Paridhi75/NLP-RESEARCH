"""
Model Training CLI for Legal Meaning Preservation.
Trains and verifies:
1. Model 1: Cross-Lingual TF-IDF Baseline
2. Model 2: Sentence-BERT
3. Model 3: Proposed MiniLM-L12 Neuro-Symbolic Model
"""
import sys
import os

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from src.minilm_l12 import run_training_pipeline


def main():
    res = run_training_pipeline()
    print("\n--- Execution Summary ---")
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
