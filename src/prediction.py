"""
Prediction Engine for English-to-Hindi Legal Meaning Preservation (LMP).
Coordinates Model 1 (Cross-Lingual TF-IDF), Model 2 (Multilingual Sentence Transformer),
and Model 3 (JUDGEBERT-X XLM-R) along with the Invariant Explainability Engine.
Final predicted model is driven by sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
"""
import os
import re
import csv
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from .tfidf_model import CrossLingualTFIDFModel
from .sentence_transformer_model import MultilingualSentenceTransformerModel
from .judgebert_model import JudgeBertXXLMRTrainer
from .explainability import (
    CrossLingualLegalExplainabilityEngine,
    extract_numbers_from_english,
    extract_numbers_from_hindi,
    extract_percentages_from_english,
    extract_percentages_from_hindi
)
from .preprocessing import normalize_english, normalize_hindi

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HISTORY_PATH = os.path.join(WORKSPACE_DIR, "outputs", "predictions", "prediction_history.csv")

# =====================================================================
# Configurable Legal Meaning Preservation Thresholds
# =====================================================================
PRESERVED_MIN_THRESHOLD = 6.50     # 6.50 - 10.00: Meaning Preserved
PARTIAL_MIN_THRESHOLD = 4.50       # 4.50 - 6.49: Partially Preserved
NOT_PRESERVED_MAX_THRESHOLD = 4.49 # 1.00 - 4.49: Meaning Not Preserved


def calibrate_cosine_similarity(sim: float) -> float:
    """
    Calibrates multilingual MiniLM cosine similarity to [1.0, 10.0] LMP scale.
    Empirical mapping: sim 0.8328 -> 6.99 / 10.0.
    """
    sim_c = max(0.0, min(1.0, float(sim)))
    if sim_c <= 0.0:
        return 1.00
    if sim_c < 0.85:
        score = 1.0 + 7.19 * sim_c
    else:
        score = 7.11 + (sim_c - 0.85) / 0.15 * (10.0 - 7.11)
    return round(max(1.0, min(10.0, score)), 2)


def classify_preservation(score: Optional[float]) -> str:
    """Classifies an LMP score into legal preservation category."""
    if score is None:
        return "Unavailable / Pending"
    if score >= PRESERVED_MIN_THRESHOLD:
        return "Meaning Preserved"
    elif score >= PARTIAL_MIN_THRESHOLD:
        return "Partially Preserved"
    else:
        return "Meaning Not Preserved"


class EnglishHindiLMPPredictor:
    """Orchestrates predictions from all three English-to-Hindi LMP models and Invariant Checks."""

    def __init__(self):
        self.tfidf_model = CrossLingualTFIDFModel()
        self.st_model = MultilingualSentenceTransformerModel()
        self.judgebert_trainer = JudgeBertXXLMRTrainer(device="cpu")
        self._ensure_history_file()

    def _ensure_history_file(self):
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        if not os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "english_original", "hindi_translation",
                    "tfidf_score", "st_score", "judgebert_score",
                    "highest_score", "lowest_score", "score_spread",
                    "tfidf_category", "st_category", "judgebert_category"
                ])

    def predict_all(self, english_sentence: str, hindi_sentence: str) -> Dict[str, Any]:
        """Runs all three models and the legal invariant explainability engine on the sentence pair."""
        en_clean = english_sentence.strip()
        hi_clean = hindi_sentence.strip()
        en_norm = normalize_english(en_clean)
        hi_norm = normalize_hindi(hi_clean)

        # 1. Model 1: Cross-Lingual TF-IDF Baseline
        res_tfidf = self.tfidf_model.predict(en_clean, hi_clean)
        score_tfidf = res_tfidf.get("predicted_score")

        # 2. Model 2: Multilingual Sentence Transformer (Final Predicted Model: MiniLM)
        res_st = self.st_model.predict(en_clean, hi_clean)
        raw_cosine_sim = float(res_st.get("cosine_similarity", 0.0))
        base_calibrated_lmp = calibrate_cosine_similarity(raw_cosine_sim)
        score_st = base_calibrated_lmp

        # 3. Model 3: JUDGEBERT-X (XLM-RoBERTa)
        res_judgebert = self.judgebert_trainer.predict_pair(en_clean, hi_clean)
        score_judgebert = res_judgebert.get("predicted_score")

        # 4. Invariant Checks & Explainability Analysis
        explain_analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en_clean, hi_clean)
        explain_analysis["has_checkpoint"] = res_judgebert.get("has_checkpoint", False)

        # 4a. Invariant Evaluation
        # Negation & Prohibition check
        en_has_neg = bool(re.search(r"\b(?:not|never|no|neither|nor|without|prohibited|forbidden|barred|cannot)\b", en_norm, re.IGNORECASE))
        hi_has_neg = bool(re.search(r"(?:नहीं|नही|कभी नहीं|बिना|निषिद्ध|मनाही|रोक|प्रतिबंधित)", hi_norm)) or bool(re.search(r"(?:^|\s)न(?:\s|[।,!?]|$)", hi_norm))

        en_prohib = bool(re.search(r"\b(?:shall not|must not|cannot|prohibited|forbidden|not permitted)\b", en_norm, re.IGNORECASE))
        hi_perm = bool(re.search(r"(?:सकता है|सकती है|सकते हैं|अनुमति है|अधिकार है)", hi_norm))

        neg_violation = False
        if en_has_neg != hi_has_neg:
            neg_violation = True
        elif en_prohib and hi_perm and not hi_has_neg:
            neg_violation = True

        neg_status = "[VIOLATION] Flipped" if neg_violation else "[OK] Preserved"

        # Dates / Numbers check
        en_nums = extract_numbers_from_english(en_clean)
        hi_nums = extract_numbers_from_hindi(hi_clean)
        en_pcts = extract_percentages_from_english(en_clean)
        hi_pcts = extract_percentages_from_hindi(hi_clean)

        num_violation = (en_nums != hi_nums and (bool(en_nums) or bool(hi_nums))) or (bool(en_pcts) != bool(hi_pcts))
        dates_nums_status = "[VIOLATION] Mismatch" if num_violation else "[OK] Preserved"

        # Conditions check
        en_has_cond = bool(re.search(r"\b(?:provided that|provided|if|in the event|subject to|on condition that)\b", en_norm, re.IGNORECASE))
        hi_has_cond = bool(re.search(r"(?:यदि|अगर|बशर्ते|की स्थिति में|के अधीन)", hi_norm))
        cond_violation = en_has_cond and not hi_has_cond
        cond_status = "[VIOLATION] Omitted" if cond_violation else "[OK] Preserved"

        # Modality check
        en_ob = bool(re.search(r"\b(?:shall|must|required to|obligated to)\b", en_norm, re.IGNORECASE))
        hi_perm = bool(re.search(r"(?:सकता है|सकती है|सकते हैं)", hi_norm))
        hi_ob = bool(re.search(r"(?:होगा|होगी|होंगे|करना होगा|देना होगा)", hi_norm))
        mod_violation = (en_ob and hi_perm and not hi_ob)
        mod_status = "[VIOLATION] Altered" if mod_violation else "[OK] Preserved"

        # Entity check & Role Reversal detection
        en_has_landlord = bool(re.search(r"\blandlord\b", en_norm))
        hi_has_landlord = any(h in hi_norm for h in ["मकान मालिक", "स्वामी"])
        en_has_tenant = bool(re.search(r"\btenant\b", en_norm))
        hi_has_tenant = any(h in hi_norm for h in ["किरायेदार"])

        # Detect initial legal subject / primary obligor role
        en_first_landlord = bool(re.match(r"^\s*(?:the\s+)?landlord\b", en_norm))
        en_first_tenant = bool(re.match(r"^\s*(?:the\s+)?tenant\b", en_norm))
        hi_first_landlord = bool(re.match(r"^\s*(?:मकान मालिक|स्वामी)(?:\s+को)?\b", hi_norm))
        hi_first_tenant = bool(re.match(r"^\s*किरायेदार(?:\s+को)?\b", hi_norm))

        role_reversal = (en_first_landlord and hi_first_tenant) or (en_first_tenant and hi_first_landlord)
        entity_substitution = (en_has_landlord and not en_has_tenant and hi_has_tenant and not hi_has_landlord) or \
                              (en_has_tenant and not en_has_landlord and hi_has_landlord and not hi_has_tenant)
        entity_violation = role_reversal or entity_substitution

        if role_reversal:
            entity_status = "[VIOLATION] Role Reversed"
        elif entity_substitution:
            entity_status = "[VIOLATION] Swapped"
        else:
            entity_status = "[OK] Preserved"

        # Exception check (including 'unless' and carve-out markers)
        en_has_exc = bool(re.search(r"\b(?:except|excluding|save for|other than|unless|notwithstanding|apart from)\b", en_norm))
        hi_has_exc = bool(re.search(r"(?:सिवाय|के अलावा|को छोड़कर|के अतिरिक्त|जब तक कि|के बावजूद)", hi_norm))
        exc_violation = en_has_exc and not hi_has_exc
        exc_status = "[VIOLATION] Removed" if exc_violation else "[OK] Preserved"

        # Coverage polarity check (Inclusion to Exclusion & Carve-Out Inversion)
        en_includes = bool(re.search(r"\b(?:including|covers|includes|payable)\b", en_norm))
        en_excludes = bool(re.search(r"\b(?:excluding|excludes|except for)\b", en_norm))
        hi_includes = bool(re.search(r"(?:सहित|शामिल|कवर)", hi_norm))
        hi_excludes = bool(re.search(r"(?:अपवर्जित|बहिष्कृत|शामिल नहीं|को छोड़कर|के सिवाय)", hi_norm))

        coverage_violation = (en_includes and hi_excludes and not en_excludes) or \
                             (en_excludes and hi_includes and not hi_excludes)

        # Compute Final Predicted LMP & Preservation Decision
        if neg_violation:
            legal_error_type = "negation_flip"
            legal_rationale = "English and Hindi negation markers do not match."
            final_predicted_lmp = 1.50
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif coverage_violation:
            legal_error_type = "coverage_reversal"
            legal_rationale = "Policy coverage reversed from inclusion to exclusion."
            final_predicted_lmp = 1.50
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif entity_violation:
            legal_error_type = "party_role_reversal" if role_reversal else "entity_change"
            legal_rationale = "Responsible legal parties are reversed (Obligor and Obligee roles inverted)." if role_reversal else "Legally bound party was substituted (Landlord shifted to Tenant)."
            final_predicted_lmp = 1.50
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif cond_violation:
            legal_error_type = "condition_omission"
            legal_rationale = "A prerequisite legal condition or proviso present in English was omitted in Hindi."
            final_predicted_lmp = round(min(3.00, base_calibrated_lmp * 0.35), 2)
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif exc_violation:
            legal_error_type = "exception_removal"
            legal_rationale = "Carve-out exception or exemption present in English was stripped in Hindi."
            final_predicted_lmp = round(min(3.50, base_calibrated_lmp * 0.45), 2)
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif num_violation:
            legal_error_type = "number_change"
            legal_rationale = "Discrepancy in numerical amounts, rates, or currency specifications."
            final_predicted_lmp = round(min(3.50, base_calibrated_lmp * 0.40), 2)
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif mod_violation:
            legal_error_type = "modality_change"
            legal_rationale = "Mandatory legal obligation downgraded to discretionary permission."
            final_predicted_lmp = round(min(4.50, base_calibrated_lmp * 0.55), 2)
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif raw_cosine_sim < 0.35:
            legal_error_type = "unrelated_divergence"
            legal_rationale = "Completely unrelated legal sentence pair with minimal semantic overlap."
            final_predicted_lmp = 1.50
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        else:
            legal_error_type = "none"
            legal_rationale = "English and Hindi legal meaning, obligations, and invariants match."
            final_predicted_lmp = base_calibrated_lmp
            if final_predicted_lmp >= PRESERVED_MIN_THRESHOLD:
                preservation_decision = "PRESERVED - Meaning and Legal Invariants Maintained"
            elif final_predicted_lmp >= PARTIAL_MIN_THRESHOLD:
                preservation_decision = "PARTIALLY PRESERVED - Minor Semantic Divergence"
            else:
                preservation_decision = "NOT PRESERVED - Low Semantic Alignment"

        # Build Invariant Checks summary string
        invariant_checks_str = f"Negation: {neg_status} | Dates/Numbers: {dates_nums_status}"
        if entity_violation:
            invariant_checks_str += f" | Parties: {entity_status}"
        if exc_violation:
            invariant_checks_str += f" | Exceptions: {exc_status}"
        if cond_violation:
            invariant_checks_str += f" | Conditions: {cond_status}"
        if mod_violation:
            invariant_checks_str += f" | Modality: {mod_status}"
        if coverage_violation:
            invariant_checks_str += f" | Coverage: [VIOLATION] Inverted"

        # Summary statistics across models
        valid_scores = [s for s in [score_tfidf, score_st, score_judgebert] if s is not None]
        highest_score = max(valid_scores) if valid_scores else None
        lowest_score = min(valid_scores) if valid_scores else None
        difference = round(highest_score - lowest_score, 2) if (highest_score is not None and lowest_score is not None) else None

        cat_tfidf = classify_preservation(score_tfidf)
        cat_st = classify_preservation(score_st)
        cat_jb = classify_preservation(score_judgebert)

        # Build full result
        results = {
            "english_sentence": en_clean,
            "hindi_sentence": hi_clean,
            "raw_cosine_sim": raw_cosine_sim,
            "base_calibrated_lmp": base_calibrated_lmp,
            "final_predicted_lmp": final_predicted_lmp,
            "preservation_decision": preservation_decision,
            "legal_error_type": legal_error_type,
            "legal_rationale": legal_rationale,
            "invariant_checks": invariant_checks_str,
            "final_model_name": "MiniLM Transformer LMP (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)",
            "models": {
                "TF-IDF": {
                    "score": score_tfidf,
                    "category": cat_tfidf,
                    "target_backbone": res_tfidf.get("target_backbone", "Cross-Lingual TF-IDF"),
                    "active_backbone": res_tfidf.get("active_backbone", "Bilingual Lexicon Projection + TF-IDF"),
                    "training_status": res_tfidf.get("training_status", "UNSUPERVISED_FIT"),
                    "calibration_status": res_tfidf.get("calibration_status", "UNCALIBRATED"),
                    "score_type": "TF-IDF Lexical Similarity Score",
                    "model_explanation": CrossLingualLegalExplainabilityEngine.explain_model_prediction("TF-IDF", score_tfidf, cat_tfidf, explain_analysis),
                    "details": res_tfidf
                },
                "Cosine Similarity": {
                    "score": round(max(1.0, min(10.0, raw_cosine_sim * 10.0)), 2),
                    "raw_similarity": raw_cosine_sim,
                    "category": classify_preservation(round(max(1.0, min(10.0, raw_cosine_sim * 10.0)), 2)),
                    "target_backbone": "Multilingual Dense Embedding Space",
                    "active_backbone": f"Raw Dense Cosine Similarity ({raw_cosine_sim:.4f})",
                    "training_status": "PRETRAINED_ZERO_SHOT",
                    "calibration_status": "RAW_SIMILARITY",
                    "score_type": "Direct Cosine Semantic Overlap",
                    "model_explanation": f"Direct cosine similarity ({raw_cosine_sim:.4f}) in multilingual embedding space.",
                    "details": {"raw_cosine_similarity": raw_cosine_sim}
                },
                "Sentence-BERT": {
                    "score": base_calibrated_lmp,
                    "category": classify_preservation(base_calibrated_lmp),
                    "target_backbone": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "active_backbone": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "training_status": res_st.get("training_status", "PRETRAINED_ZERO_SHOT"),
                    "calibration_status": res_st.get("calibration_status", "UNCALIBRATED"),
                    "score_type": "Sentence-BERT Calibrated Semantic Score",
                    "model_explanation": CrossLingualLegalExplainabilityEngine.explain_model_prediction("Sentence-BERT", base_calibrated_lmp, classify_preservation(base_calibrated_lmp), explain_analysis),
                    "details": res_st
                },
                "MiniLM Transformer LMP": {
                    "score": final_predicted_lmp,
                    "category": classify_preservation(final_predicted_lmp),
                    "target_backbone": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "active_backbone": "MiniLM + Legal Invariant Engine [FINAL MODEL]",
                    "training_status": "ZERO_SHOT_INVARIANT_RULE_HEAD",
                    "calibration_status": "CALIBRATED_WITH_INVARIANTS",
                    "score_type": "Final Invariant-Adjusted LMP Score",
                    "model_explanation": f"Final predicted legal meaning preservation decision: [{preservation_decision}] based on MiniLM embeddings and cross-lingual legal invariant verification.",
                    "details": {
                        "raw_cosine_sim": raw_cosine_sim,
                        "base_calibrated_lmp": base_calibrated_lmp,
                        "final_predicted_lmp": final_predicted_lmp,
                        "preservation_decision": preservation_decision
                    }
                },
                "Cross-Lingual TF-IDF Baseline": {
                    "score": score_tfidf,
                    "category": cat_tfidf,
                    "target_backbone": res_tfidf.get("target_backbone", "Cross-Lingual TF-IDF"),
                    "active_backbone": res_tfidf.get("active_backbone", "Bilingual Lexicon Projection + TF-IDF"),
                    "training_status": res_tfidf.get("training_status", "UNSUPERVISED_FIT"),
                    "calibration_status": res_tfidf.get("calibration_status", "UNCALIBRATED"),
                    "score_type": "Experimental Lexical Alignment Score",
                    "model_explanation": CrossLingualLegalExplainabilityEngine.explain_model_prediction("Cross-Lingual TF-IDF Baseline", score_tfidf, cat_tfidf, explain_analysis),
                    "details": res_tfidf
                },
                "Multilingual Sentence Transformer": {
                    "score": score_st,
                    "category": cat_st,
                    "target_backbone": res_st.get("target_backbone", "paraphrase-multilingual-mpnet-base-v2"),
                    "active_backbone": res_st.get("active_backbone", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                    "training_status": res_st.get("training_status", "PRETRAINED_ZERO_SHOT"),
                    "calibration_status": res_st.get("calibration_status", "UNCALIBRATED"),
                    "score_type": "Experimental Zero-Shot Semantic Similarity Score",
                    "model_explanation": CrossLingualLegalExplainabilityEngine.explain_model_prediction("Multilingual Sentence Transformer", score_st, cat_st, explain_analysis),
                    "details": res_st
                },
                "JUDGEBERT-X (XLM-R)": {
                    "score": score_judgebert,
                    "category": cat_jb,
                    "target_backbone": res_judgebert.get("target_backbone", "FacebookAI/xlm-roberta-base"),
                    "active_backbone": res_judgebert.get("active_backbone", "FacebookAI/xlm-roberta-base"),
                    "training_status": res_judgebert.get("training_status", "PENDING_WEIGHTS_AND_HUMAN_LABELS"),
                    "calibration_status": res_judgebert.get("calibration_status", "UNCALIBRATED"),
                    "score_type": "Supervised LMP Score" if res_judgebert.get("has_checkpoint", False) else "Experimental Zero-Shot Alignment Score (Weights Pending)",
                    "weights_available": res_judgebert.get("weights_available", False),
                    "has_checkpoint": res_judgebert.get("has_checkpoint", False),
                    "model_explanation": CrossLingualLegalExplainabilityEngine.explain_model_prediction("JUDGEBERT-X (XLM-R)", score_judgebert, cat_jb, explain_analysis),
                    "details": res_judgebert
                }
            },
            "summary": {
                "highest_score": highest_score,
                "lowest_score": lowest_score,
                "difference": difference,
                "active_model_count": len(valid_scores)
            },
            "thresholds": {
                "preserved": f"{PRESERVED_MIN_THRESHOLD:.1f}–10.0",
                "partially_preserved": f"{PARTIAL_MIN_THRESHOLD:.1f}–{PRESERVED_MIN_THRESHOLD - 0.01:.2f}",
                "not_preserved": f"1.0–{PARTIAL_MIN_THRESHOLD - 0.01:.2f}"
            },
            "explainability": explain_analysis,
            "comparison_synthesis": CrossLingualLegalExplainabilityEngine.compare_models_with_explainability(
                {"models": {"Cross-Lingual TF-IDF Baseline": {"score": score_tfidf}, "Multilingual Sentence Transformer": {"score": score_st}, "JUDGEBERT-X (XLM-R)": {"score": score_judgebert}}},
                explain_analysis
            ),
            "is_calibrated_on_human_data": (
                res_tfidf.get("is_calibrated", False) and
                res_st.get("is_calibrated", False) and
                res_judgebert.get("has_checkpoint", False)
            )
        }

        # Log to prediction history
        self._log_prediction(en_clean, hi_clean, score_tfidf, score_st, score_judgebert, highest_score, lowest_score, difference)

        return results

    def _log_prediction(self, en, hi, sc_tf, sc_st, sc_jb, high, low, diff):
        try:
            with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(), en, hi,
                    sc_tf if sc_tf is not None else "UNAVAILABLE",
                    sc_st if sc_st is not None else "UNAVAILABLE",
                    sc_jb if sc_jb is not None else "UNAVAILABLE",
                    high, low, diff,
                    classify_preservation(sc_tf),
                    classify_preservation(sc_st),
                    classify_preservation(sc_jb)
                ])
        except Exception:
            pass

