"""
Legal Meaning Preservation Package.
Strictly implements three models:
1. CrossLingualTFIDFModel (TF-IDF + Cosine Similarity)
2. MultilingualSentenceTransformerModel (Sentence-BERT)
3. MiniLML12Model (Proposed MiniLM-L12 Neuro-Symbolic Model)
"""
from .data_loader import EnglishHindiDataLoader, EnglishHindiDataLoader as DataLoader
from .preprocessing import normalize_english, normalize_hindi, tokenize_english, tokenize_hindi
from .tfidf_model import CrossLingualTFIDFModel
from .sentence_transformer_model import MultilingualSentenceTransformerModel
from .minilm_l12 import MiniLML12Model, run_training_pipeline
from .prediction import EnglishHindiLMPPredictor
from .explainability import CrossLingualLegalExplainabilityEngine
from .evaluation import EnglishHindiEvaluator

__all__ = [
    "EnglishHindiDataLoader",
    "DataLoader",
    "normalize_english",
    "normalize_hindi",
    "tokenize_english",
    "tokenize_hindi",
    "CrossLingualTFIDFModel",
    "MultilingualSentenceTransformerModel",
    "MiniLML12Model",
    "EnglishHindiLMPPredictor",
    "CrossLingualLegalExplainabilityEngine",
    "EnglishHindiEvaluator",
    "run_training_pipeline",
]
