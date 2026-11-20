"""
Comprehensive Automated Unit and System Test Suite for English-to-Hindi Legal Meaning Preservation (LMP).
Verifies:
1. All three models accept English original and Hindi translation.
2. Output scores are strictly bounded within [1.0, 10.0] or report UNAVAILABLE if weights missing.
3. Truthful backbone reporting: MiniLM is reported as MiniLM, never falsely as MPNet;
   JUDGEBERT-X target backbone is FacebookAI/xlm-roberta-base, never silently substituted with MiniLM.
4. XLM-R architecture: Tokenizer, 768-dim config, pair representation [u; v; |u-v|] (2304-dim).
5. Missing checkpoints and uncalibrated models are handled safely without crashing.
6. Zero-human-label training protection: training pipeline cleanly refuses to fabricate ground truth.
7. Zero cross-monolingual contamination: 100% of pairs are strictly English-to-Hindi.
8. Zero clause overlap across train, val, and test splits.
9-18. All 10 legal explainability categories with exact statuses (DETECTED, NEEDS REVIEW, NOT DETECTED).
19. Devanagari Unicode and numeral handling.
20. Important test case (Tenant rent with 5% penalty proviso) identifying condition omission, penalty omission, and percentage omission with overlap notes.
21. Terminal prediction pipeline structure.
"""
import os
import sys
import unittest
import pandas as pd
import unicodedata
import torch

# Enforce offline mode in sandbox
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE_DIR)

from src.data_loader import EnglishHindiDataLoader
from src.tfidf_model import CrossLingualTFIDFModel
from src.sentence_transformer_model import MultilingualSentenceTransformerModel
from src.judgebert_model import JudgeBertXXLMRTrainer, JudgeBertXXLMR
from src.prediction import EnglishHindiLMPPredictor, classify_preservation
from src.explainability import (
    CrossLingualLegalExplainabilityEngine,
    devanagari_to_arabic_digits,
    extract_numbers_from_english,
    extract_numbers_from_hindi,
    extract_percentages_from_english,
    extract_percentages_from_hindi
)
from src.preprocessing import normalize_english, normalize_hindi, tokenize_english, tokenize_hindi


class TestEnglishHindiLMP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = EnglishHindiLMPPredictor()
        cls.loader = EnglishHindiDataLoader()

        cls.sample_en = "The insurer shall not be liable for any loss or damage arising from the use of the vehicle by an unlicensed person."
        cls.sample_hi_faithful = "बीमाकर्ता बिना लाइसेंस वाले व्यक्ति द्वारा वाहन के उपयोग से उत्पन्न किसी भी हानि के लिए उत्तरदायी नहीं होगा।"
        cls.sample_hi_perturbed = "किरायेदार हर महीने मकान मालिक को समय पर किराया देगा।"

    # -----------------------------------------------------------------
    # 1. Model Input and Prediction Acceptance
    # -----------------------------------------------------------------
    def test_01_models_accept_english_hindi_pairs(self):
        """Verify that all three models accept an English sentence and a Hindi sentence."""
        res = self.predictor.predict_all(self.sample_en, self.sample_hi_faithful)
        self.assertIn("models", res)
        self.assertIn("Cross-Lingual TF-IDF Baseline", res["models"])
        self.assertIn("Multilingual Sentence Transformer", res["models"])
        self.assertIn("JUDGEBERT-X (XLM-R)", res["models"])

        sc_tf = res["models"]["Cross-Lingual TF-IDF Baseline"]["score"]
        sc_st = res["models"]["Multilingual Sentence Transformer"]["score"]
        print(f"[TEST 1] TF-IDF: {sc_tf}, Sentence Transformer: {sc_st}")
        self.assertIsNotNone(sc_tf)
        self.assertIsNotNone(sc_st)

    # -----------------------------------------------------------------
    # 2. Output Score Bounds
    # -----------------------------------------------------------------
    def test_02_output_score_bounds(self):
        """Verify that all numeric scores are strictly between 1.0 and 10.0."""
        res = self.predictor.predict_all(self.sample_en, self.sample_hi_perturbed)
        for m_name, info in res["models"].items():
            sc = info["score"]
            if sc is not None:
                self.assertGreaterEqual(sc, 1.0, f"{m_name} score {sc} < 1.0")
                self.assertLessEqual(sc, 10.0, f"{m_name} score {sc} > 10.0")
                print(f"[TEST 2] {m_name} score {sc} within [1.0, 10.0] bounds.")
            else:
                print(f"[TEST 2] {m_name} score truthfully reported as UNAVAILABLE (weights not cached offline).")

    # -----------------------------------------------------------------
    # 3. Truthful Backbone Reporting & No Silent MiniLM Substitution
    # -----------------------------------------------------------------
    def test_03_truthful_backbone_and_no_silent_minilm_substitution(self):
        """
        Verify that:
        1. Sentence Transformer truthfully identifies whether MPNet or MiniLM is loaded.
        2. JUDGEBERT-X target backbone is FacebookAI/xlm-roberta-base and NEVER silently substitutes MiniLM.
        """
        st_info = self.predictor.st_model.predict(self.sample_en, self.sample_hi_faithful)
        self.assertIn("active_backbone", st_info)
        # Verify MiniLM is identified as MiniLM
        if self.predictor.st_model.is_fallback_active:
            self.assertIn("MiniLM", st_info["active_backbone"])
            self.assertEqual(st_info["loaded_backbone_short"], "MiniLM-L12-v2")
            print("[TEST 3] Sentence Transformer truthfully reports active MiniLM fallback.")

        jb_info = self.predictor.judgebert_trainer.predict_pair(self.sample_en, self.sample_hi_faithful)
        self.assertEqual(jb_info["target_backbone"], "FacebookAI/xlm-roberta-base")
        # Ensure it does NOT report MiniLM as XLM-R
        self.assertNotIn("MiniLM", jb_info["active_backbone"])
        if not jb_info["weights_available"]:
            self.assertIsNone(jb_info["predicted_score"])
            self.assertEqual(jb_info["mode"], "WEIGHTS_UNAVAILABLE_OFFLINE")
            print("[TEST 3] JUDGEBERT-X truthfully reports weights unavailable rather than substituting MiniLM.")

    # -----------------------------------------------------------------
    # 4. XLM-RoBERTa Architecture and Pair Representation [u; v; |u-v|]
    # -----------------------------------------------------------------
    def test_04_xlmr_architecture_and_pair_representation(self):
        """Verify XLM-RoBERTa 768-dim config and 2304-dim Siamese pair representation."""
        model = JudgeBertXXLMR()
        self.assertEqual(model.encoder_name, "FacebookAI/xlm-roberta-base")
        self.assertEqual(model.hidden_dim, 768)

        # Verify regression head dimension: 3 * 768 = 2304 -> 128 -> 1
        first_layer = model.regression_head[0]
        self.assertEqual(first_layer.in_features, 2304)
        self.assertEqual(first_layer.out_features, 128)

        # Test forward pass with synthetic 768-dim tensors
        u = torch.randn(2, 768)
        v = torch.randn(2, 768)
        diff = torch.abs(u - v)
        pair = torch.cat([u, v, diff], dim=1)
        self.assertEqual(pair.shape, (2, 2304))

        out = model.regression_head(pair)
        scaled = 1.0 + 9.0 * out.squeeze(-1)
        self.assertTrue((scaled >= 1.0).all() and (scaled <= 10.0).all())
        print("[TEST 4] XLM-RoBERTa pair representation [u; v; |u-v|] (2304-dim) and regression head verified.")

    # -----------------------------------------------------------------
    # 5. Missing Checkpoint Safety
    # -----------------------------------------------------------------
    def test_05_missing_checkpoint_safety(self):
        """Verify that missing supervised checkpoints are handled safely without crashing."""
        trainer = JudgeBertXXLMRTrainer(device="cpu")
        res = trainer.predict_pair(self.sample_en, self.sample_hi_faithful)
        self.assertFalse(res["has_checkpoint"])
        self.assertIn(res["mode"], ["ZERO_SHOT_EXPERIMENTAL_MODE", "WEIGHTS_UNAVAILABLE_OFFLINE"])
        print(f"[TEST 5] Missing supervised checkpoint handled safely: {res['mode']}.")

    # -----------------------------------------------------------------
    # 6. Zero Human Label Training Protection
    # -----------------------------------------------------------------
    def test_06_zero_human_label_training_protection(self):
        """Verify training pipeline aborts cleanly when 0 genuine human pairs exist."""
        from src.training import run_training_pipeline
        res = run_training_pipeline()
        self.assertEqual(res["status"], "AWAITING_HUMAN_ANNOTATIONS")
        self.assertEqual(res["genuine_annotated_samples"], 0)
        self.assertFalse(res["judgebert_xlmr_trained"])
        print("[TEST 6] Zero-human-label training protection verified (refuses to train on fake labels).")

    # -----------------------------------------------------------------
    # 7. Zero Cross-Monolingual Contamination
    # -----------------------------------------------------------------
    def test_07_zero_cross_monolingual_contamination(self):
        """Verify that NO English-English or Hindi-Hindi pairs contaminate the aligned dataset."""
        df_aligned = self.loader.load_aligned_dataset()
        self.assertEqual(len(df_aligned), 160)

        for _, r in df_aligned.iterrows():
            en_text = r["english_original"]
            hi_text = r["hindi_translation"]

            has_latin = any("a" <= ch.lower() <= "z" for ch in en_text)
            self.assertTrue(has_latin, f"English text does not contain Latin script: {en_text}")

            has_devanagari = any(0x0900 <= ord(ch) <= 0x097F for ch in hi_text)
            self.assertTrue(has_devanagari, f"Hindi text does not contain Devanagari script: {hi_text}")

        print("[TEST 7] Verified 100% of aligned pairs are strictly English-to-Hindi with zero monolingual leakage.")

    # -----------------------------------------------------------------
    # 8. Zero Clause Overlap Across Splits
    # -----------------------------------------------------------------
    def test_08_splits_zero_clause_leakage(self):
        """Verify zero clause leakage across train, val, and test splits."""
        train, val, test = self.loader.load_splits()
        c_train = set(train["pair_id"].apply(lambda x: str(x).split("-")[0]))
        c_val = set(val["pair_id"].apply(lambda x: str(x).split("-")[0]))
        c_test = set(test["pair_id"].apply(lambda x: str(x).split("-")[0]))

        self.assertEqual(len(c_train.intersection(c_val)), 0)
        self.assertEqual(len(c_train.intersection(c_test)), 0)
        self.assertEqual(len(c_val.intersection(c_test)), 0)
        print("[TEST 8] Zero clause overlap across train, val, and test partitions verified.")

    # -----------------------------------------------------------------
    # 9. Explainability: Number and Currency Changes
    # -----------------------------------------------------------------
    def test_09_explainability_number_and_currency_changes(self):
        """Verify detection of changed numerical quantities and currencies."""
        en = "The deductible of Rs. 5,000 shall apply to each claim."
        hi = "प्रत्येक दावे पर 50,000 रुपये की कटौती लागू होगी।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["number_changes"], 0)
        self.assertTrue(any(f["category"] == "NUMBER CHANGE" and f["status"] == "DETECTED" for f in analysis["findings"]))
        print("[TEST 9] NUMBER AND CURRENCY CHANGE detected successfully (5,000 vs 50,000).")

    # -----------------------------------------------------------------
    # 10. Explainability: Negation Reversal
    # -----------------------------------------------------------------
    def test_10_explainability_negation_reversal(self):
        """Verify detection of prohibition flipped to permission."""
        en = "The tenant must not sublet the property."
        hi = "किरायेदार संपत्ति किराए पर दे सकता है।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["negation_changes"], 0)
        self.assertTrue(any(f["category"] == "NEGATION CHANGE" and f["status"] == "DETECTED" for f in analysis["findings"]))
        print("[TEST 10] NEGATION REVERSAL detected successfully (prohibition to affirmative permission).")

    # -----------------------------------------------------------------
    # 11. Explainability: Modality Degradation
    # -----------------------------------------------------------------
    def test_11_explainability_modality_degradation(self):
        """Verify mandatory obligation degraded to discretionary permission."""
        en = "The insured person shall undergo a medical examination."
        hi = "बीमित व्यक्ति मेडिकल जांच करवा सकता है।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["modality_changes"], 0)
        self.assertTrue(any(f["category"] == "MODALITY CHANGE" and f["status"] == "DETECTED" for f in analysis["findings"]))
        print("[TEST 11] MODALITY DEGRADATION detected successfully (shall to sakta hai).")

    # -----------------------------------------------------------------
    # 12. Explainability: Entity or Party Changes
    # -----------------------------------------------------------------
    def test_12_explainability_entity_party_change(self):
        """Verify substitution of legally bound parties."""
        en = "The landlord shall maintain the roof and external walls."
        hi = "किरायेदार छत और बाहरी दीवारों का रखरखाव करेगा।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["entity_changes"], 0)
        self.assertTrue(any(f["category"] == "ENTITY CHANGE" and f["status"] == "DETECTED" for f in analysis["findings"]))
        print("[TEST 12] ENTITY OR PARTY CHANGE detected successfully (Landlord to Tenant).")

    # -----------------------------------------------------------------
    # 13. Explainability: Condition Omission
    # -----------------------------------------------------------------
    def test_13_explainability_condition_omission(self):
        """Verify omitted condition marker."""
        en = "Ambulance charges shall be reimbursed, provided hospitalisation is admissible."
        hi = "एम्बुलेंस शुल्क की प्रतिपूर्ति की जाएगी।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["condition_omissions"], 0)
        self.assertTrue(any(f["category"] == "CONDITION OMISSION" and f["status"] == "DETECTED" for f in analysis["findings"]))
        print("[TEST 13] CONDITION OMISSION detected successfully (provided that dropped).")

    # -----------------------------------------------------------------
    # 14. Explainability: Exception Removal
    # -----------------------------------------------------------------
    def test_14_explainability_exception_removal(self):
        """Verify stripped carve-out or exception."""
        en = "Hospital expenses are payable except for day-care procedures in Annexure II."
        hi = "अस्पताल के खर्च देय हैं।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["exception_removals"], 0)
        self.assertTrue(any(f["category"] == "EXCEPTION REMOVAL" and f["status"] == "DETECTED" for f in analysis["findings"]))
        print("[TEST 14] EXCEPTION REMOVAL detected successfully (except carve-out stripped).")

    # -----------------------------------------------------------------
    # 15. Explainability: Scope Divergence
    # -----------------------------------------------------------------
    def test_15_explainability_scope_divergence(self):
        """Verify restrictive scope divergence."""
        en = "The policy covers third-party damage solely arising from insured vehicle operation."
        hi = "पॉलिसी वाहन संचालन से उत्पन्न तीसरे पक्ष के नुकसान को कवर करती है।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["scope_changes"], 0)
        self.assertTrue(any(f["category"] == "SCOPE CHANGE" and f["status"] == "NEEDS REVIEW" for f in analysis["findings"]))
        print("[TEST 15] SCOPE DIVERGENCE detected successfully (solely stripped).")

    # -----------------------------------------------------------------
    # 16. Explainability: Temporal Changes
    # -----------------------------------------------------------------
    def test_16_explainability_temporal_change(self):
        """Verify temporal deadline inversion."""
        en = "Notice must be given before the commencement of repairs."
        hi = "मरम्मत शुरू होने के बाद नोटिस दिया जाना चाहिए।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["temporal_changes"], 0)
        self.assertTrue(any(f["category"] == "DATE AND TEMPORAL CHANGE" and f["status"] == "DETECTED" for f in analysis["findings"]))
        print("[TEST 16] DATE AND TEMPORAL CHANGE detected successfully (before vs after).")

    # -----------------------------------------------------------------
    # 17. Explainability: Hallucinated Information
    # -----------------------------------------------------------------
    def test_17_explainability_hallucinated_information(self):
        """Verify detection of unauthorized penalty added in Hindi."""
        en = "The tenant shall pay monthly rent on the first day of each month."
        hi = "किरायेदार पहली तारीख को किराया देगा अन्यथा भारी जुर्माना लगेगा।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["hallucinated_information"], 0)
        self.assertTrue(any(f["category"] == "HALLUCINATED INFORMATION" and f["status"] == "DETECTED" for f in analysis["findings"]))
        print("[TEST 17] HALLUCINATED INFORMATION detected successfully (unauthorized जुर्माना in Hindi).")

    # -----------------------------------------------------------------
    # 18. Explainability: Missing Information / Severe Compression
    # -----------------------------------------------------------------
    def test_18_explainability_missing_information_compression(self):
        """Verify severe length compression and omitted information detection."""
        en = "The tenant shall maintain the premises, repair internal defects, insure personal property, and indemnify the landlord against all claims arising from negligence."
        hi = "किरायेदार परिसर में रहेगा।"
        analysis = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)
        self.assertGreater(analysis["counts"]["omissions"], 0)
        self.assertTrue(any(f["category"] == "OMISSION" for f in analysis["findings"]))
        print("[TEST 18] MISSING INFORMATION / SEVERE COMPRESSION detected successfully.")

    # -----------------------------------------------------------------
    # 19. Devanagari Unicode and Numeral Handling
    # -----------------------------------------------------------------
    def test_19_devanagari_unicode_handling(self):
        """Verify proper Devanagari numeral conversion, NFC normalization, and danda handling."""
        hi_dev_digits = "किरायेदार प्रत्येक माह की ५ तारीख को १५००० रुपये देगा।"
        conv = devanagari_to_arabic_digits(hi_dev_digits)
        self.assertIn("5", conv)
        self.assertIn("15000", conv)

        nums = extract_numbers_from_hindi(hi_dev_digits)
        self.assertIn(5, nums)
        self.assertIn(15000, nums)

        norm_hi = normalize_hindi("किराया  देगा |")
        self.assertTrue(norm_hi.endswith("।"))
        self.assertEqual(unicodedata.is_normalized("NFC", norm_hi), True)
        print("[TEST 19] Devanagari Unicode, numerals (५ -> 5), and danda normalization verified.")

    # -----------------------------------------------------------------
    # 20. Important Test Case: Tenant Rent with 5% Penalty Proviso
    # -----------------------------------------------------------------
    def test_20_important_test_case_synthesis(self):
        """
        Test case from Section 3:
        English: The tenant shall pay a monthly rent of Rs 15,000 before the 5th day of each month, provided that late payment shall incur a penalty of 5%.
        Hindi: किरायेदार प्रत्येक माह की 5 तारीख से पहले 15,000 रुपये मासिक किराया देगा।

        Verifies:
        1. Condition omission: 'provided that' detected.
        2. Penalty obligation omission: 'late payment shall incur a penalty' detected.
        3. Missing 5% penalty rate detected.
        4. Overlap note explained so findings are not counted blindly without context.
        """
        en = "The tenant shall pay a monthly rent of Rs 15,000 before the 5th day of each month, provided that late payment shall incur a penalty of 5%."
        hi = "किरायेदार प्रत्येक माह की 5 तारीख से पहले 15,000 रुपये मासिक किराया देगा।"

        res = CrossLingualLegalExplainabilityEngine.analyze_english_hindi_pair(en, hi)

        # 1. Condition omission
        has_cond = any(f["category"] == "CONDITION OMISSION" and f["status"] == "DETECTED" for f in res["findings"])
        self.assertTrue(has_cond, "Failed to detect missing 'provided that' condition.")

        # 2. Penalty consequence omission
        has_penalty_omit = any(f["category"] == "OMISSION" and f["status"] == "DETECTED" for f in res["findings"])
        self.assertTrue(has_penalty_omit, "Failed to detect missing penalty obligation.")

        # 3. Missing 5% penalty rate
        has_pct_omit = any(f["category"] == "NUMBER CHANGE" and "5%" in f.get("english", "") for f in res["findings"])
        self.assertTrue(has_pct_omit, "Failed to detect missing 5% penalty rate.")

        # 4. Overlap notes present
        has_overlap_note = any(f.get("overlap_note") is not None for f in res["findings"])
        self.assertTrue(has_overlap_note, "Missing overlap explanation for related omissions.")

        print("[TEST 20] Section 3 Important Test Case verified (condition, penalty obligation, 5% rate, and overlap explained).")

    # -----------------------------------------------------------------
    # 21. Terminal Prediction Pipeline Structure
    # -----------------------------------------------------------------
    def test_21_terminal_prediction_pipeline_structure(self):
        """Verify full prediction dictionary structure for terminal output."""
        res = self.predictor.predict_all(self.sample_en, self.sample_hi_faithful)
        self.assertIn("english_sentence", res)
        self.assertIn("hindi_sentence", res)
        self.assertIn("models", res)
        self.assertIn("summary", res)
        self.assertIn("thresholds", res)
        self.assertIn("explainability", res)
        self.assertIn("comparison_synthesis", res)
        print("[TEST 21] Prediction pipeline dictionary structure verified.")


if __name__ == "__main__":
    unittest.main()
