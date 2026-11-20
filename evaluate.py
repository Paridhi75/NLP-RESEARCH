"""
Master Evaluation CLI for English-to-Hindi Legal Meaning Preservation (LMP).
Evaluates all three models on the exact same English-Hindi test pairs.
Calculates ground truth metrics when human labels are present,
and contrastive discriminative metrics when human annotations are pending.
"""
import sys
import os
import pandas as pd

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from src.evaluation import EnglishHindiEvaluator


def main():
    print("======================================================================")
    print("   JUDGEBERT-X: ENGLISH-TO-HINDI BENCHMARK EVALUATION SUITE           ")
    print("======================================================================")
    evaluator = EnglishHindiEvaluator()
    summary = evaluator.run_evaluation()

    print("\n======================================================================")
    print("                    EVALUATION BENCHMARK RESULTS                      ")
    print("======================================================================")
    metrics_file = os.path.join(WORKSPACE_DIR, "outputs", "reports", "english_hindi_evaluation.csv")
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        print(df.to_string(index=False))

    if not summary["has_ground_truth"]:
        print("\n----------------------------------------------------------------------")
        print("ℹ️  NOTE: English-to-Hindi evaluation performed in honest zero-shot mode.")
        print("   Human ground-truth LMP scores are currently pending.")
        print("   Contrastive discrimination verifies that faithful legal translations")
        print("   receive higher LMP scores than legally perturbed/corrupted translations.")
        print("----------------------------------------------------------------------")


if __name__ == "__main__":
    main()
