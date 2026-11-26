"""
Automated Test Suite for Legal Meaning Preservation (LMP) Framework.
Validates strictly the three models:
1. CrossLingualTFIDFModel (Model 1)
2. MultilingualSentenceTransformerModel (Model 2)
3. MiniLML12Model (Model 3 - Proposed MiniLM-L12 Neuro-Symbolic Model)
"""
import os
import sys
import unittest
import pandas as pd

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_DIR)

from src.data_loader import EnglishHindiDataLoader
from src.tfidf_model import CrossLingualTFIDFModel
from src.sentence_transformer_model import MultilingualSentenceTransformerModel
from src.minilm_l12 import MiniLML12Model
from src.prediction import EnglishHindiLMPPredictor, classify_preservation
from src.explainability import CrossLingualLegalExplainabilityEngine


class TestMiniLML12Framework(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = EnglishHindiLMPPredictor()
        cls.loader = EnglishHindiDataLoader()

        cls.sample_en = "The Tenant shall pay monthly rent of Rs. 25,000 on or before the fifth day."
        cls.sample_hi_faithful = "किरायेदार पांचवीं तारीख को या उससे पहले 25,000 रुपये का मासिक किराया देगा।"
        cls.sample_hi_perturbed = "किरायेदार घरेलू पालतू जानवर रख सकता है।"

    def test_model_1_tfidf(self):
        """Tests Model 1: Cross-Lingual TF-IDF Baseline."""
        model = CrossLingualTFIDFModel()
        model.fit(["The Tenant shall pay rent."], ["किरायेदार किराया देगा।"])
        res = model.predict(self.sample_en, self.sample_hi_faithful)
        self.assertIn("predicted_score", res)
        self.assertGreaterEqual(res["predicted_score"], 1.0)
        self.assertLessEqual(res["predicted_score"], 10.0)

    def test_model_2_sentence_transformer(self):
        """Tests Model 2: Multilingual Sentence Transformer (Sentence-BERT)."""
        model = MultilingualSentenceTransformerModel()
        res = model.predict(self.sample_en, self.sample_hi_faithful)
        self.assertIn("cosine_similarity", res)
        self.assertGreaterEqual(res["cosine_similarity"], -1.0)
        self.assertLessEqual(res["cosine_similarity"], 1.0)

    def test_model_3_minilm_l12(self):
        """Tests Model 3: Proposed MiniLM-L12 Neuro-Symbolic Model."""
        model = MiniLML12Model()
        res = model.predict_pair(self.sample_en, self.sample_hi_faithful)
        self.assertIn("predicted_score", res)
        self.assertGreaterEqual(res["predicted_score"], 1.0)
        self.assertLessEqual(res["predicted_score"], 10.0)

    def test_unified_predictor(self):
        """Tests the unified prediction engine coordinating all three models."""
        res = self.predictor.predict_all(self.sample_en, self.sample_hi_faithful)
        models = res["models"]
        self.assertIn("TF-IDF + Cosine Similarity", models)
        self.assertIn("Sentence-BERT", models)
        self.assertIn("Proposed MiniLM-L12", models)
        self.assertEqual(len(models), 3)

        # Final predicted score must be in [1.0, 10.0]
        final_sc = res["final_predicted_lmp"]
        self.assertGreaterEqual(final_sc, 1.0)
        self.assertLessEqual(final_sc, 10.0)

    def test_invariant_explainability_engine(self):
        """Tests legal invariant analysis across key categories."""
        # Modality shift: shall -> may (होगा -> सकता है)
        en_ob = "The Tenant shall vacate the premises."
        hi_perm = "किरायेदार परिसर को खाली कर सकता है।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en_ob, hi_perm)
        self.assertIn("counts", analysis)
        self.assertGreaterEqual(analysis["counts"]["modality_changes"], 1)


if __name__ == "__main__":
    unittest.main()
