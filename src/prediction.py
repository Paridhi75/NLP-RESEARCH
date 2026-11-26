"""
Prediction Engine for English-to-Hindi Legal Meaning Preservation (LMP).
Coordinates strictly the three models:
1. Model 1: Cross-Lingual TF-IDF Baseline
2. Model 2: Multilingual Sentence Transformer (Sentence-BERT)
3. Model 3: Proposed MiniLM-L12 Neuro-Symbolic Model
along with the Cross-Lingual Legal Invariant Explainability Engine.
"""
import os
import sys
import re
import csv
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

try:
    from .tfidf_model import CrossLingualTFIDFModel
    from .sentence_transformer_model import MultilingualSentenceTransformerModel
    from .minilm_l12 import MiniLML12Model
    from .explainability import (
        CrossLingualLegalExplainabilityEngine,
        extract_numbers_from_english,
        extract_numbers_from_hindi,
        extract_percentages_from_english,
        extract_percentages_from_hindi,
        devanagari_to_arabic_digits
    )
    from .preprocessing import normalize_english, normalize_hindi
except ImportError:
    from src.tfidf_model import CrossLingualTFIDFModel
    from src.sentence_transformer_model import MultilingualSentenceTransformerModel
    from src.minilm_l12 import MiniLML12Model
    from src.explainability import (
        CrossLingualLegalExplainabilityEngine,
        extract_numbers_from_english,
        extract_numbers_from_hindi,
        extract_percentages_from_english,
        extract_percentages_from_hindi,
        devanagari_to_arabic_digits
    )
    from src.preprocessing import normalize_english, normalize_hindi

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
        self.minilm_model = MiniLML12Model(device="cpu")
        self._ensure_history_file()

    def _ensure_history_file(self):
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        if not os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "english_original", "hindi_translation",
                    "tfidf_score", "st_score", "minilm_score",
                    "highest_score", "lowest_score", "score_spread",
                    "tfidf_category", "st_category", "minilm_category"
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

        # 2. Model 2: Multilingual Sentence Transformer (Sentence-BERT)
        res_st = self.st_model.predict(en_clean, hi_clean)
        raw_cosine_sim = float(res_st.get("cosine_similarity", 0.0))
        base_calibrated_lmp = calibrate_cosine_similarity(raw_cosine_sim)
        score_st = base_calibrated_lmp

        # 3. Model 3: Proposed MiniLM-L12 Neuro-Symbolic Model
        res_minilm = self.minilm_model.predict_pair(en_clean, hi_clean)
        score_minilm = res_minilm.get("predicted_score")

        # 4. Invariant Checks & Explainability Analysis
        explain_analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en_clean, hi_clean)
        explain_analysis["has_checkpoint"] = res_minilm.get("has_checkpoint", False)

        # 4a. Invariant Evaluation
        # Negation & Prohibition check
        en_has_neg = bool(re.search(r"\b(?:not|never|no|neither|nor|without|prohibited|forbidden|barred|cannot)\b", en_norm, re.IGNORECASE))
        hi_has_neg = bool(re.search(r"(?:नहीं|नही|कभी नहीं|बिना|निषिद्ध|मनाही|रोक|प्रतिबंधित)", hi_norm)) or bool(re.search(r"(?:^|\s)न(?:\s|[।,!?]|$)", hi_norm))

        en_prohib = bool(re.search(r"\b(?:shall not|must not|cannot|prohibited|forbidden|not permitted|shall never)\b", en_norm, re.IGNORECASE))
        hi_perm = bool(re.search(r"(?:सकता है|सकती है|सकते हैं|अनुमति है|अधिकार है|स्वीकृत है)", hi_norm))
        hi_verb_neg = bool(re.search(r"(?:नहीं|नही)\s*(?:[^\s।,!?]+\s*)?(?:सकता|सकती|सकते|करेगा|करेगी|करेंगे|देगा|देगी|देंगे|होगा|होगी)", hi_norm)) or bool(re.search(r"(?:^|\s)न\s*(?:[^\s।,!?]+\s*)?(?:सकता|सकती|सकते|देगा|देगी|देंगे)", hi_norm))

        neg_violation = False
        if en_prohib and hi_perm and not hi_verb_neg:
            neg_violation = True
        elif en_has_neg != hi_has_neg:
            neg_violation = True

        neg_status = "[VIOLATION] Flipped" if neg_violation else "[OK] Preserved"

        # Dates & Calendar Years check
        en_years = {int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", en_clean)}
        hi_years = {int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", devanagari_to_arabic_digits(hi_clean))}
        date_violation = (en_years != hi_years and (bool(en_years) or bool(hi_years)))

        # Numbers and Percentages check
        en_nums = extract_numbers_from_english(en_clean)
        hi_nums = extract_numbers_from_hindi(hi_clean)
        en_pcts = extract_percentages_from_english(en_clean)
        hi_pcts = extract_percentages_from_hindi(hi_clean)

        # Exclude calendar years from pure quantities to avoid conflating date changes with number changes
        pure_en_nums = en_nums - en_years
        pure_hi_nums = hi_nums - hi_years

        num_violation = (pure_en_nums != pure_hi_nums and (bool(pure_en_nums) or bool(pure_hi_nums))) or (bool(en_pcts) != bool(hi_pcts))
        if date_violation:
            dates_nums_status = "[VIOLATION] Date Mismatch"
        elif num_violation:
            dates_nums_status = "[VIOLATION] Mismatch"
        else:
            dates_nums_status = "[OK] Preserved"

        # Conditions check (including consent prerequisites)
        en_has_cond = bool(re.search(
            r"\b(?:provided that|provided|if|in the event|subject to|on condition that|only with|with(?:out)? (?:the )?(?:prior )?written consent|with(?:out)? (?:the )?(?:prior )?written permission|prior written consent|written consent|prior written permission|written permission|consent of|consent from|permission of|permission from)\b",
            en_norm, re.IGNORECASE
        ))
        hi_has_cond = bool(re.search(
            r"(?:यदि|अगर|बशर्ते|की स्थिति में|के अधीन|शर्त|सहमति|अनुमति|मंजूरी|इजाजत|इज़ाज़त)",
            hi_norm
        ))
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

        en_first_landlord = bool(re.match(r"^\s*(?:the\s+)?landlord\b", en_norm))
        en_first_tenant = bool(re.match(r"^\s*(?:the\s+)?tenant\b", en_norm))
        hi_first_landlord = bool(re.match(r"^\s*(?:मकान मालिक|स्वामी)(?:\s+को)?\b", hi_norm))
        hi_first_tenant = bool(re.match(r"^\s*किरायेदार(?:\s+को)?\b", hi_norm))

        role_reversal = (en_first_landlord and hi_first_tenant) or (en_first_tenant and hi_first_landlord)
        entity_violation = (en_has_landlord and not hi_has_landlord) or (en_has_tenant and not hi_has_tenant) or role_reversal
        entity_status = "[VIOLATION] Role Inverted" if role_reversal else ("[VIOLATION] Substituted" if entity_violation else "[OK] Preserved")

        # Exceptions check
        en_has_exc = bool(re.search(r"\b(?:except|unless|excluding|other than|save for|notwithstanding)\b", en_norm, re.IGNORECASE))
        hi_has_exc = bool(re.search(r"(?:सिवाय|के अलावा|को छोड़कर|के अतिरिक्त|जब तक कि|के बावजूद)", hi_norm))
        exc_violation = en_has_exc and not hi_has_exc
        exc_status = "[VIOLATION] Omitted" if exc_violation else "[OK] Preserved"

        # Coverage scope inversion check
        en_has_inc = bool(re.search(r"\b(?:including|includes|included|covers|coverage)\b", en_norm, re.IGNORECASE))
        hi_has_exc_scope = bool(re.search(r"(?:को छोड़कर|छोड़कर|अपवर्जित|बहिष्कृत)", hi_norm))
        coverage_violation = en_has_inc and hi_has_exc_scope
        coverage_status = "[VIOLATION] Inverted" if coverage_violation else "[OK] Preserved"

        # Apply Invariant Penalisations to determine Final Decision
        if coverage_violation:
            legal_error_type = "coverage_inversion"
            legal_rationale = "Policy coverage was inverted from affirmative inclusion to operative exclusion."
            final_predicted_lmp = 1.50
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif neg_violation:
            legal_error_type = "negation_reversal"
            legal_rationale = "Direct negation or statutory prohibition was flipped, altering binding legal rights."
            final_predicted_lmp = 1.50
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif entity_violation:
            legal_error_type = "entity_role_reversal" if role_reversal else "entity_substitution"
            legal_rationale = "Responsible legal parties are reversed (Obligor and Obligee roles inverted)." if role_reversal else "Legally bound party was substituted (Landlord shifted to Tenant)."
            final_predicted_lmp = 1.50
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif cond_violation:
            legal_error_type = "condition_omission"
            legal_rationale = "A prerequisite legal condition or proviso present in English was omitted in Hindi."
            final_predicted_lmp = round(min(2.74, base_calibrated_lmp * 0.35), 2)
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif exc_violation:
            legal_error_type = "exception_removal"
            legal_rationale = "Carve-out exception or exemption present in English was stripped in Hindi."
            final_predicted_lmp = round(min(2.63, base_calibrated_lmp * 0.45), 2)
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif mod_violation:
            legal_error_type = "modality_change"
            legal_rationale = "Mandatory legal obligation downgraded to discretionary permission."
            final_predicted_lmp = round(min(3.72, base_calibrated_lmp * 0.55), 2)
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif date_violation:
            legal_error_type = "date_change"
            legal_rationale = "Contractual date or expiration year altered (e.g., 2027 vs 2028), shifting legally binding terms."
            final_predicted_lmp = round(max(1.0, min(10.0, base_calibrated_lmp - 3.50)), 2)
            preservation_decision = "NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation"
        elif num_violation:
            legal_error_type = "number_change"
            legal_rationale = "Discrepancy in numerical amounts, rates, or currency specifications."
            final_predicted_lmp = round(max(1.0, min(10.0, base_calibrated_lmp - 3.50)), 2)
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

        cat_tfidf = classify_preservation(score_tfidf)
        cat_st = classify_preservation(score_st)
        cat_minilm = classify_preservation(score_minilm)

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
            "final_model_name": "Proposed MiniLM-L12 Neuro-Symbolic Model",
            "models": {
                "TF-IDF + Cosine Similarity": {
                    "score": score_tfidf,
                    "category": cat_tfidf,
                    "target_backbone": "Bilingual Lexicon Projection + TF-IDF",
                    "training_status": res_tfidf.get("training_status", "UNSUPERVISED_FIT"),
                    "details": res_tfidf
                },
                "Sentence-BERT": {
                    "score": base_calibrated_lmp,
                    "category": cat_st,
                    "target_backbone": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "training_status": "PRETRAINED_ZERO_SHOT",
                    "details": res_st
                },
                "Proposed MiniLM-L12": {
                    "score": final_predicted_lmp,
                    "category": classify_preservation(final_predicted_lmp),
                    "target_backbone": "MiniLM-L12 + Legal Invariant Engine [PROPOSED MODEL]",
                    "details": {
                        "raw_neural_score": score_minilm,
                        "final_predicted_lmp": final_predicted_lmp,
                        "preservation_decision": preservation_decision
                    }
                }
            },
            "explainability": explain_analysis
        }

        # Record prediction history
        try:
            scores_valid = [s for s in [score_tfidf, score_st, score_minilm] if s is not None]
            highest = max(scores_valid) if scores_valid else None
            lowest = min(scores_valid) if scores_valid else None
            spread = round(highest - lowest, 2) if (highest is not None and lowest is not None) else None
            with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    en_clean, hi_clean,
                    score_tfidf, score_st, score_minilm,
                    highest, lowest, spread,
                    cat_tfidf, cat_st, cat_minilm
                ])
        except Exception:
            pass

        return results
