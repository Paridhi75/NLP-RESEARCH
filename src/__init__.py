"""
JUDGEBERT-X English-to-Hindi Legal Meaning Preservation Package.
"""
from .data_loader import EnglishHindiDataLoader, EnglishHindiDataLoader as DataLoader
from .preprocessing import normalize_english, normalize_hindi, tokenize_english, tokenize_hindi
from .tfidf_model import CrossLingualTFIDFModel
from .sentence_transformer_model import MultilingualSentenceTransformerModel
from .judgebert_model import JudgeBertXXLMR, JudgeBertXXLMRTrainer
from .prediction import EnglishHindiLMPPredictor
from .explainability import CrossLingualLegalExplainabilityEngine
from .evaluation import EnglishHindiEvaluator
from .training import run_training_pipeline

__all__ = [
    "EnglishHindiDataLoader",
    "DataLoader",
    "normalize_english",
    "normalize_hindi",
    "tokenize_english",
    "tokenize_hindi",
    "CrossLingualTFIDFModel",
    "MultilingualSentenceTransformerModel",
    "JudgeBertXXLMR",
    "JudgeBertXXLMRTrainer",
    "EnglishHindiLMPPredictor",
    "CrossLingualLegalExplainabilityEngine",
    "EnglishHindiEvaluator",
    "run_training_pipeline",
]
