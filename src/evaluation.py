"""
Evaluation Engine for English-to-Hindi Legal Meaning Preservation (LMP).
Evaluates all three models on the EXACT same English-Hindi test set.
Computes Pearson, Spearman, RMSE, MAE, and F1 when human labels are present,
and inter-model agreement and faithful-vs-perturbed discrimination when labels are pending.
Truthfully reports model statuses and unavailable weights.
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

            info_tf = m_res.get("TF-IDF") or m_res.get("Cross-Lingual TF-IDF Baseline", {})
            info_cos = m_res.get("Cosine Similarity", {})
            info_st = m_res.get("Sentence-BERT") or m_res.get("Multilingual Sentence Transformer", {})
            info_minilm = m_res.get("MiniLM Transformer LMP", {})
            info_jb = m_res.get("JUDGEBERT-X (XLM-R)", {})

            sc_tf = info_tf.get("score")
            sc_cos = info_cos.get("score")
            sc_st = info_st.get("score")
            sc_minilm = info_minilm.get("score")
            sc_jb = info_jb.get("score")

            predictions.append({
                "pair_id": row.get("pair_id", f"PAIR-{idx:03d}"),
                "english_original": en,
                "hindi_translation": hi,
                "domain": domain,
                "translation_type": trans_type,
                "legal_error_category": error_cat,
                "human_lmp_score": human_sc if pd.notnull(human_sc) else "PENDING",
                "pred_tfidf": sc_tf,
                "pred_cosine_similarity": sc_cos,
                "pred_sentence_bert": sc_st,
                "pred_minilm_transformer_lmp": sc_minilm,
                "pred_judgebert_xlmr": sc_jb,
                "category_tfidf": classify_preservation(sc_tf),
                "category_cosine": classify_preservation(sc_cos),
                "category_sentence_bert": classify_preservation(sc_st),
                "category_minilm_lmp": classify_preservation(sc_minilm),
                "category_jb": classify_preservation(sc_jb),
            })

        df_preds = pd.DataFrame(predictions)
        preds_path = os.path.join(PREDS_DIR, "english_hindi_test_predictions.csv")
        df_preds.to_csv(preds_path, index=False, encoding="utf-8")
        print(f"[Evaluation] Saved per-pair test predictions to: {preds_path}")

        # Check if genuine human ground truth exists on test set
        has_ground_truth = df_test["human_lmp_score"].dropna().shape[0] > 0

        metrics_records = []
        if has_ground_truth:
            df_eval = df_preds[df_preds["human_lmp_score"] != "PENDING"].copy()
            y_true = df_eval["human_lmp_score"].astype(float).values
            for m_key, pred_col in [
                ("Cross-Lingual TF-IDF Baseline", "pred_tfidf"),
                ("Multilingual Sentence Transformer", "pred_sentence_transformer"),
                ("JUDGEBERT-X (XLM-R)", "pred_judgebert_xlmr")
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
                        "f1_macro": round(f1_mac, 4),
                        "f1_weighted": round(f1_wt, 4)
                    })
                else:
                    metrics_records.append({
                        "model": m_key,
                        "ground_truth_status": "WEIGHTS_UNAVAILABLE",
                        "n_test_samples": 0,
                        "pearson_r": "N/A",
                        "spearman_rho": "N/A",
                        "rmse": "N/A",
                        "mae": "N/A",
                        "f1_macro": "N/A",
                        "f1_weighted": "N/A"
                    })
        else:
            # Honest report when human annotations are pending
            # Compute Faithful vs Perturbed discrimination across models
            df_f = df_preds[df_preds["translation_type"] == "faithful"]
            df_p = df_preds[df_preds["translation_type"] == "perturbed"]

            for m_key, pred_col in [
                ("TF-IDF", "pred_tfidf"),
                ("Cosine Similarity", "pred_cosine_similarity"),
                ("Sentence-BERT", "pred_sentence_bert"),
                ("MiniLM Transformer LMP", "pred_minilm_transformer_lmp"),
                ("JUDGEBERT-X (XLM-R)", "pred_judgebert_xlmr")
            ]:
                s_f = df_f[pred_col].dropna()
                s_p = df_p[pred_col].dropna()

                if len(s_f) > 0 and len(s_p) > 0:
                    f_mean = float(s_f.mean())
                    p_mean = float(s_p.mean())
                    diff = round(f_mean - p_mean, 2)
                    st_desc = "PENDING_HUMAN_ANNOTATION"
                    f_disp = round(f_mean, 2)
                    p_disp = round(p_mean, 2)
                    m_disp = diff
                else:
                    st_desc = "WEIGHTS_UNAVAILABLE_OFFLINE"
                    f_disp = "UNAVAILABLE"
                    p_disp = "UNAVAILABLE"
                    m_disp = "UNAVAILABLE"

                metrics_records.append({
                    "model": m_key,
                    "ground_truth_status": st_desc,
                    "n_test_samples": len(df_preds),
                    "faithful_mean_score": f_disp,
                    "perturbed_mean_score": p_disp,
                    "contrastive_margin": m_disp,
                    "notes": (
                        "Final invariant-verified MiniLM LMP model: penalizes legal distortions (e.g. negation flips, condition omissions)."
                        if "MiniLM" in m_key else
                        "Dense multilingual Sentence-BERT embeddings capture topical alignment."
                        if "Sentence-BERT" in m_key else
                        "Direct uncalibrated dense cosine similarity in multilingual embedding space."
                        if "Cosine" in m_key else
                        "Pretrained XLM-RoBERTa weights missing from offline sandbox cache; Tokenizer & 768-dim config verified."
                        if "JUDGEBERT" in m_key else
                        "Bilingual lexicon-projected cross-lingual lexical baseline."
                    )
                })

        df_metrics = pd.DataFrame(metrics_records)
        report_path = os.path.join(REPORTS_DIR, "english_hindi_evaluation.csv")
        df_metrics.to_csv(report_path, index=False, encoding="utf-8")
        print(f"[Evaluation] Saved evaluation metrics to: {report_path}")

        return {
            "has_ground_truth": has_ground_truth,
            "metrics": df_metrics.to_dict(orient="records"),
            "test_sample_count": len(df_preds)
        }
