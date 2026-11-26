"""
Proposed MiniLM-L12 Neuro-Symbolic Framework and Training Module (renamed from training.py).
Implements Model 3:
- Shared Multilingual MiniLM-L12 Encoder (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, 384-dim)
- Sentence representations: English u, Candidate v, Element-wise difference |u - v|
- Tri-vector pair representation: [u; v; |u - v|] in R^1152
- Multi-layer regression head: LayerNorm, GELU, Dropout (0.2), Sigmoid scaled to [1.0, 10.0]
- Supervised and zero-shot training pipeline for Legal Meaning Preservation
"""
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

import sys
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

try:
    from .data_loader import EnglishHindiDataLoader
    from .tfidf_model import CrossLingualTFIDFModel
    from .sentence_transformer_model import MultilingualSentenceTransformerModel
except ImportError:
    from src.data_loader import EnglishHindiDataLoader
    from src.tfidf_model import CrossLingualTFIDFModel
    from src.sentence_transformer_model import MultilingualSentenceTransformerModel

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models", "proposed_minilm")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "pytorch_model.bin")
CONFIG_PATH = os.path.join(MODELS_DIR, "config.json")
TRAIN_LOG_PATH = os.path.join(MODELS_DIR, "train_log.json")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MiniLMRegressionHead(nn.Module):
    """
    Neural regression prediction head mapping [u; v; |u - v|] in R^1152 to [1.0, 10.0] scale.
    """
    def __init__(self, in_features: int = 1152, hidden1: int = 256, hidden2: int = 64, dropout_prob: float = 0.20):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.act2 = nn.GELU()
        
        self.fc3 = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.act1(self.ln1(self.fc1(x)))
        z1 = self.dropout(z1)
        z2 = self.act2(self.ln2(self.fc2(z1)))
        norm_score = self.sigmoid(self.fc3(z2))
        lmp_score = 1.0 + 9.0 * norm_score
        return lmp_score


class MiniLML12Model:
    """
    Model 3: Proposed MiniLM-L12 Neuro-Symbolic Framework.
    Extracts dense 384-dimensional sentence vectors using multilingual MiniLM-L12,
    forms the 1152-dimensional tri-vector [u; v; |u - v|], and applies the neural regression head.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.encoder: Optional[SentenceTransformer] = None
        self.regression_head = MiniLMRegressionHead().to(self.device)
        self.has_checkpoint = False
        self._load_checkpoint_if_exists()

    def _ensure_encoder(self) -> SentenceTransformer:
        if self.encoder is None:
            self.encoder = SentenceTransformer(MODEL_NAME, local_files_only=True)
        return self.encoder

    def _load_checkpoint_if_exists(self):
        if os.path.exists(CHECKPOINT_PATH):
            try:
                self.regression_head.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))
                self.regression_head.eval()
                self.has_checkpoint = True
            except Exception:
                self.has_checkpoint = False

    def compute_pair_features(self, embs_u: np.ndarray, embs_v: np.ndarray) -> np.ndarray:
        diff = np.abs(embs_u - embs_v)
        return np.concatenate([embs_u, embs_v, diff], axis=-1)

    def predict_pair(self, text_en: str, text_cand: str) -> Dict[str, Any]:
        """Computes dense representations and predicts LMP score for a sentence pair."""
        encoder = self._ensure_encoder()
        u = encoder.encode([text_en], convert_to_numpy=True)
        v = encoder.encode([text_cand], convert_to_numpy=True)
        feat = self.compute_pair_features(u, v)

        tensor_x = torch.tensor(feat, dtype=torch.float32, device=self.device)
        self.regression_head.eval()
        with torch.no_grad():
            raw_score = float(self.regression_head(tensor_x).cpu().numpy().flatten()[0])

        return {
            "model_name": "Proposed MiniLM-L12",
            "predicted_score": round(raw_score, 4),
            "target_backbone": MODEL_NAME,
            "has_checkpoint": self.has_checkpoint
        }

    def predict_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
        encoder = self._ensure_encoder()
        origs = [p[0] for p in pairs]
        cands = [p[1] for p in pairs]

        embs_u = encoder.encode(origs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        embs_v = encoder.encode(cands, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        features = self.compute_pair_features(embs_u, embs_v)

        tensor_x = torch.tensor(features, dtype=torch.float32, device=self.device)
        self.regression_head.eval()
        with torch.no_grad():
            preds = self.regression_head(tensor_x).cpu().numpy().flatten()
        return [round(float(p), 4) for p in preds]

    def fit(self, train_pairs: List[Tuple[str, str]], train_scores: List[float], epochs: int = 5, lr: float = 2e-5) -> Dict[str, Any]:
        set_seed(42)
        encoder = self._ensure_encoder()
        origs = [p[0] for p in train_pairs]
        cands = [p[1] for p in train_pairs]

        embs_u = encoder.encode(origs, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        embs_v = encoder.encode(cands, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        features = self.compute_pair_features(embs_u, embs_v)

        x_train = torch.tensor(features, dtype=torch.float32)
        y_train = torch.tensor(train_scores, dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
        optimizer = torch.optim.AdamW(self.regression_head.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()

        history = []
        for epoch in range(1, epochs + 1):
            self.regression_head.train()
            running_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.regression_head(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(bx)
            epoch_loss = running_loss / len(x_train)
            history.append(round(epoch_loss, 4))

        # Save model checkpoint
        torch.save(self.regression_head.state_dict(), CHECKPOINT_PATH)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "Proposed MiniLM-L12 Neuro-Symbolic Model",
                "backbone": MODEL_NAME,
                "in_features": 1152,
                "hidden1": 256,
                "hidden2": 64,
                "dropout": 0.20
            }, f, indent=2)

        with open(TRAIN_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump({"train_loss": history, "epochs": epochs}, f, indent=2)

        self.has_checkpoint = True
        return {"final_loss": history[-1], "history": history}


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes training pipeline across all three models:
    Model 1: Cross-Lingual TF-IDF Baseline
    Model 2: Multilingual Sentence Transformer Calibration
    Model 3: Proposed MiniLM-L12 Neuro-Symbolic Model
    """
    loader = EnglishHindiDataLoader()
    stats = loader.get_statistics()
    df_aligned = loader.load_aligned_dataset()

    print("======================================================================")
    print("      THREE-MODEL TRAINING PIPELINE (TF-IDF, S-BERT, MINILM-L12)      ")
    print("======================================================================")
    print(f"Total Aligned Pairs:                 {stats['total_aligned_pairs']}")
    print(f"Genuine Human-Annotated Pairs:       {stats['genuine_human_annotated_pairs']}")
    print(f"Pending Annotation Pairs:            {stats['pending_annotation_pairs']}")
    print("======================================================================")

    # 1. Train Model 1: TF-IDF
    print("\n[Model 1] Fitting Cross-Lingual TF-IDF Baseline...")
    tfidf_model = CrossLingualTFIDFModel()
    tfidf_model.fit(
        en_corpus=list(df_aligned["english_original"]),
        hi_corpus=list(df_aligned["hindi_translation"])
    )
    print("  -> Cross-Lingual TF-IDF baseline fitted successfully.")

    # 2. Train Model 2: Sentence-BERT
    print("\n[Model 2] Checking Sentence Transformer calibration data...")
    st_model = MultilingualSentenceTransformerModel()
    df_annotated = loader.load_annotated_dataset()
    if len(df_annotated) >= 5:
        pairs = list(zip(df_annotated["english_original"], df_annotated["hindi_translation"]))
        scores = list(df_annotated["human_lmp_score"].astype(float))
        st_model.fit_calibration(pairs, scores)
        print("  -> Sentence Transformer calibrated on human scores.")
    else:
        print("  -> Sentence-BERT operating in zero-shot cross-lingual calibration mode.")

    # 3. Train Model 3: Proposed MiniLM-L12
    print("\n[Model 3] Training Proposed MiniLM-L12 Neuro-Symbolic Model...")
    minilm_model = MiniLML12Model()
    
    # Train on available human-annotated data if present, or benchmark dataset
    en_ins_path = os.path.join(WORKSPACE_DIR, "data", "raw", "english_insurance.csv")
    en_ten_path = os.path.join(WORKSPACE_DIR, "data", "raw", "english_tenancy.xlsx")
    
    benchmark_pairs = []
    benchmark_scores = []
    if os.path.exists(en_ins_path):
        df_ins = pd.read_csv(en_ins_path)
        for _, r in df_ins.iterrows():
            orig = str(r["original_clause"]).strip()
            cand = str(r["simplified_clause"]).strip() if pd.notna(r["simplified_clause"]) else orig
            benchmark_pairs.append((orig, cand))
            benchmark_scores.append(float(r["lmp_score"]))
            
    if os.path.exists(en_ten_path):
        df_ten = pd.read_excel(en_ten_path)
        for _, r in df_ten.iterrows():
            orig = str(r["Source_Legal_Text"]).strip()
            cand = str(r["Simplified_Text"]).strip() if pd.notna(r["Simplified_Text"]) else orig
            benchmark_pairs.append((orig, cand))
            benchmark_scores.append(float(r["Human_LMP_Rating"]))

    if len(benchmark_pairs) > 0:
        res = minilm_model.fit(benchmark_pairs, benchmark_scores, epochs=5)
        print(f"  -> Proposed MiniLM-L12 trained on {len(benchmark_pairs)} benchmark pairs (Loss: {res['final_loss']:.4f}).")
        return {
            "status": "TRAINING_COMPLETE",
            "tfidf_trained": True,
            "sentence_bert_ready": True,
            "proposed_minilm_trained": True,
            "final_loss": res["final_loss"]
        }
    else:
        return {
            "status": "READY_ZERO_SHOT",
            "tfidf_trained": True,
            "sentence_bert_ready": True,
            "proposed_minilm_trained": False
        }


if __name__ == "__main__":
    print("=" * 70)
    print("  RUNNING PROPOSED MINILM-L12 NEURO-SYMBOLIC MODEL & PIPELINE")
    print("=" * 70)
    summary = run_training_pipeline()
    print("\n--- Pipeline Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n--- Testing Single Prediction ---")
    test_model = MiniLML12Model()
    sample_en = "The Tenant shall pay monthly rent of Rs. 25,000 on or before the fifth day."
    sample_hi = "किरायेदार पांचवीं तारीख को या उससे पहले 25,000 रुपये का मासिक किराया देगा।"
    pred = test_model.predict_pair(sample_en, sample_hi)
    print(f"Original Text        : {sample_en}")
    print(f"Candidate (Hindi)    : {sample_hi}")
    print(f"Predicted LMP Score  : {pred['predicted_score']:.2f} / 10.0")
    print("=" * 70)
