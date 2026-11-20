"""
Cross-Lingual TF-IDF Baseline for English-to-Hindi Legal Meaning Preservation (Model 1).
Uses bilingual lexical projection to bridge English and Hindi Devanagari vocabularies
and computes cross-lingual lexical similarity.
"""
import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import normalize_english, normalize_hindi, tokenize_english, tokenize_hindi, project_english_to_hindi_lexicon

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models", "tfidf")
os.makedirs(MODELS_DIR, exist_ok=True)


class CrossLingualTFIDFModel:
    """
    Cross-Lingual TF-IDF Baseline.
    Projects English legal terms into Hindi conceptual space using bilingual lexicon alignment,
    then computes TF-IDF cosine similarity against the Hindi translated text.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.is_fitted = False
        self.model_path = os.path.join(MODELS_DIR, "crosslingual_tfidf.pkl")
        self.calibration_slope = 9.0
        self.calibration_intercept = 1.0
        self.is_calibrated = False

    def _prepare_crosslingual_corpus(self, en_texts: List[str], hi_texts: List[str]) -> List[str]:
        """Creates joint Hindi-space training corpus."""
        corpus = []
        for en in en_texts:
            proj = project_english_to_hindi_lexicon(en)
            if proj:
                corpus.append(" ".join(proj))
        for hi in hi_texts:
            norm_hi = normalize_hindi(hi)
            if norm_hi:
                corpus.append(norm_hi)
        return corpus

    def fit(self, en_corpus: List[str], hi_corpus: List[str]):
        """Fits TF-IDF vocabulary on the aligned cross-lingual corpus."""
        joint_corpus = self._prepare_crosslingual_corpus(en_corpus, hi_corpus)
        if not joint_corpus:
            joint_corpus = ["बीमाकर्ता पॉलिसी दावा", "किरायेदार मकान मालिक परिसर किराया"]
        self.vectorizer.fit(joint_corpus)
        self.is_fitted = True
        self.save()

    def predict(self, english_text: str, hindi_text: str) -> Dict[str, Any]:
        """
        Computes cross-lingual lexical similarity between English original and Hindi translation.
        Returns predicted LMP score [1.0 - 10.0].
        """
        if not self.is_fitted:
            self.load()

        # 1. Project English into Hindi lexical space
        en_projected_tokens = project_english_to_hindi_lexicon(english_text)
        hi_tokens = tokenize_hindi(hindi_text)

        if not en_projected_tokens or not hi_tokens:
            # Fallback if no legal dictionary matches
            raw_sim = 0.10
            overlap_ratio = 0.0
        else:
            en_proj_str = " ".join(en_projected_tokens)
            hi_str = " ".join(hi_tokens)

            try:
                vecs = self.vectorizer.transform([en_proj_str, hi_str])
                cos_sim = float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
            except Exception:
                cos_sim = 0.0

            # Lexical overlap ratio (Jaccard on projected sets)
            set_en = set(en_projected_tokens)
            set_hi = set(hi_tokens)
            inter = len(set_en.intersection(set_hi))
            union = len(set_en.union(set_hi))
            overlap_ratio = float(inter / max(1, union))

            # Combined cross-lingual lexical similarity
            raw_sim = 0.60 * cos_sim + 0.40 * overlap_ratio

        raw_sim = max(0.0, min(1.0, raw_sim))
        # Scale to [1.0, 10.0]
        lmp_score = round(float(self.calibration_intercept + self.calibration_slope * raw_sim), 2)
        lmp_score = max(1.0, min(10.0, lmp_score))

        return {
            "model_name": "Cross-Lingual TF-IDF Baseline",
            "target_backbone": "Cross-Lingual TF-IDF + Bilingual Legal Lexicon",
            "active_backbone": "Bilingual Lexicon Projection + Character/Word TF-IDF",
            "predicted_score": lmp_score,
            "lexical_similarity": round(raw_sim, 4),
            "projected_tokens_matched": len(set(en_projected_tokens).intersection(set(hi_tokens))) if en_projected_tokens else 0,
            "is_calibrated": self.is_calibrated,
            "training_status": "UNSUPERVISED_FIT (No human labels)",
            "calibration_status": "CALIBRATED_ON_HUMAN_SCORES" if self.is_calibrated else "UNCALIBRATED (0 human English-Hindi pairs)",
            "score_type": "Experimental Lexical Alignment Score",
            "methodology": "Bilingual Lexicon Projection + TF-IDF Cosine Similarity",
            "limitation": "Lacks semantic compositionality across Devanagari and Latin scripts; relies strictly on dictionary coverage."
        }

    def save(self):
        """Saves fitted vectorizer."""
        with open(self.model_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self):
        """Loads fitted vectorizer or fits default fallback."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.is_fitted = True
        else:
            # Fit minimal fallback
            fallback_en = ["The insurer shall pay claim.", "The tenant shall pay monthly rent."]
            fallback_hi = ["बीमाकर्ता दावा का भुगतान करेगा।", "किरायेदार मासिक किराया अदा करेगा।"]
            self.fit(fallback_en, fallback_hi)
