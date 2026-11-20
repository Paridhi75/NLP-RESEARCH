"""
JUDGEBERT-X: English-to-Hindi Model Training CLI.
Executes supervised training of JUDGEBERT-X (XLM-RoBERTa) regression head
strictly when genuine human English-to-Hindi legal annotations are available.
"""
import sys
import os

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from src.training import run_training_pipeline


def main():
    res = run_training_pipeline()
    print("\n--- Execution Summary ---")
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
