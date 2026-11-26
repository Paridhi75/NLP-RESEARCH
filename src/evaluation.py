"""
Evaluation Engine for English-to-Hindi Legal Meaning Preservation (LMP).
Evaluates strictly the three models on the common test split:
1. Model 1: Cross-Lingual TF-IDF Baseline
2. Model 2: Multilingual Sentence Transformer (Sentence-BERT)
3. Model 3: Proposed MiniLM-L12 Neuro-Symbolic Model
Computes Pearson, Spearman, RMSE, MAE, Macro F1, and Weighted F1.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

from .data_loader import EnglishHindiDataLoader
from .prediction import EnglishHindiLMPPredictor, classify_preservation

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(WORKSPACE_DIR, "outputs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
PREDS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)


class EnglishHindiEvaluator:
    """Evaluates all three models on common English-to-Hindi test pairs."""

    def __init__(self):
        self.loader = EnglishHindiDataLoader()
        self.predictor = EnglishHindiLMPPredictor()

    def run_evaluation(self) -> Dict[str, Any]:
        """Runs evaluation across the held-out test split."""
        _, _, df_test = self.loader.load_splits()
        print(f"\n[Evaluation] Loaded {len(df_test)} common English-Hindi test pairs.")

        predictions = []
        for idx, row in df_test.iterrows():
            en = row["english_original"]
            hi = row["hindi_translation"]
            domain = row.get("domain", "Unknown")
            trans_type = row.get("translation_type", "Unknown")
            error_cat = row.get("legal_error_category", "NONE")
            human_sc = row.get("human_lmp_score", None)

            res = self.predictor.predict_all(en, hi)
            m_res = res["models"]

            info_tf = m_res.get("TF-IDF + Cosine Similarity", {})
            info_st = m_res.get("Sentence-BERT", {})
            info_minilm = m_res.get("Proposed MiniLM-L12", {})

            sc_tf = info_tf.get("score")
            sc_st = info_st.get("score")
            sc_minilm = info_minilm.get("score")

            predictions.append({
                "pair_id": row.get("pair_id", f"PAIR-{idx:03d}"),
                "english_original": en,
                "hindi_translation": hi,
                "domain": domain,
                "translation_type": trans_type,
                "legal_error_category": error_cat,
                "human_lmp_score": human_sc if pd.notnull(human_sc) else "PENDING",
                "pred_tfidf": sc_tf,
                "pred_sentence_bert": sc_st,
                "pred_proposed_minilm": sc_minilm,
                "category_tfidf": classify_preservation(sc_tf),
                "category_sentence_bert": classify_preservation(sc_st),
                "category_proposed_minilm": classify_preservation(sc_minilm),
            })

        df_preds = pd.DataFrame(predictions)
        preds_path = os.path.join(PREDS_DIR, "english_hindi_test_predictions.csv")
        df_preds.to_csv(preds_path, index=False, encoding="utf-8")
        print(f"[Evaluation] Saved per-pair test predictions to: {preds_path}")

        has_ground_truth = df_test["human_lmp_score"].dropna().shape[0] > 0
        metrics_records = []

        if has_ground_truth:
            df_eval = df_preds[df_preds["human_lmp_score"] != "PENDING"].copy()
            y_true = df_eval["human_lmp_score"].astype(float).values
            for m_key, pred_col in [
                ("TF-IDF + Cosine Similarity", "pred_tfidf"),
                ("Sentence-BERT", "pred_sentence_bert"),
                ("Proposed MiniLM-L12", "pred_proposed_minilm")
            ]:
                valid_mask = df_eval[pred_col].notnull()
                if valid_mask.sum() > 0:
                    y_p = df_eval.loc[valid_mask, pred_col].astype(float).values
                    y_t = y_true[valid_mask.values]
                    pr, _ = pearsonr(y_t, y_p) if len(y_t) > 1 else (0.0, 0.0)
                    sr, _ = spearmanr(y_t, y_p) if len(y_t) > 1 else (0.0, 0.0)
                    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
                    mae = float(mean_absolute_error(y_t, y_p))

                    true_cat = [classify_preservation(s) for s in y_t]
                    pred_cat = [classify_preservation(s) for s in y_p]
                    f1_mac = float(f1_score(true_cat, pred_cat, average="macro", zero_division=0))
                    f1_wt = float(f1_score(true_cat, pred_cat, average="weighted", zero_division=0))

                    metrics_records.append({
                        "model": m_key,
                        "ground_truth_status": "GENUINE_HUMAN_SCORES",
                        "n_test_samples": len(y_p),
                        "pearson_r": round(pr, 4),
                        "spearman_rho": round(sr, 4),
                        "rmse": round(rmse, 4),
                        "mae": round(mae, 4),
                        "macro_f1": round(f1_mac, 4),
                        "weighted_f1": round(f1_wt, 4)
                    })
        else:
            print("\n[Evaluation] Genuine human ratings on test set are pending.")
            print("[Evaluation] Reporting contrastive discrimination metrics across faithful vs corrupted test pairs:")
            faithful_mask = df_preds["translation_type"] == "faithful"
            perturbed_mask = df_preds["translation_type"] != "faithful"

            for m_key, pred_col in [
                ("TF-IDF + Cosine Similarity", "pred_tfidf"),
                ("Sentence-BERT", "pred_sentence_bert"),
                ("Proposed MiniLM-L12", "pred_proposed_minilm")
            ]:
                mean_f = df_preds.loc[faithful_mask, pred_col].mean() if faithful_mask.sum() > 0 else 0.0
                mean_p = df_preds.loc[perturbed_mask, pred_col].mean() if perturbed_mask.sum() > 0 else 0.0
                gap = round(mean_f - mean_p, 4)

                metrics_records.append({
                    "model": m_key,
                    "ground_truth_status": "ZERO_SHOT_CONTRASTIVE_EVAL",
                    "n_test_samples": len(df_preds),
                    "mean_faithful_lmp": round(float(mean_f), 4),
                    "mean_perturbed_lmp": round(float(mean_p), 4),
                    "contrastive_margin": gap,
                    "discrimination_check": "PASS" if gap > 0 else "FAIL"
                })

        df_metrics = pd.DataFrame(metrics_records)
        metrics_path = os.path.join(REPORTS_DIR, "english_hindi_evaluation.csv")
        df_metrics.to_csv(metrics_path, index=False, encoding="utf-8")
        print(f"[Evaluation] Saved evaluation benchmark report to: {metrics_path}")

        return {
            "has_ground_truth": has_ground_truth,
            "metrics": metrics_records,
            "df_metrics": df_metrics
        }
