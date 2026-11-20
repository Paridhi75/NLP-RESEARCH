"""
Training Pipeline Module for JUDGEBERT-X English-to-Hindi Meaning Preservation.
Enforces strict zero-fabrication rules: requires genuine human English-Hindi labels.
"""
import os
import json
import pandas as pd
from typing import Dict, Any

from .data_loader import EnglishHindiDataLoader
from .tfidf_model import CrossLingualTFIDFModel
from .sentence_transformer_model import MultilingualSentenceTransformerModel
from .judgebert_model import JudgeBertXXLMR, JudgeBertXXLMRTrainer

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes training pipeline across all three models.
    Strictly verifies that genuine human English-Hindi ground truth exists before training.
    """
    loader = EnglishHindiDataLoader()
    stats = loader.get_statistics()
    df_annotated = loader.load_annotated_dataset()

    print("======================================================================")
    print("        JUDGEBERT-X: ENGLISH-TO-HINDI TRAINING PIPELINE               ")
    print("======================================================================")
    print(f"Total Aligned English-Hindi Pairs:   {stats['total_aligned_pairs']}")
    print(f"Genuine Human-Annotated Pairs:       {stats['genuine_human_annotated_pairs']}")
    print(f"Pending Annotation Pairs:            {stats['pending_annotation_pairs']}")
    print("======================================================================")

    # 1. Train / Fit Model 1: Cross-Lingual TF-IDF Baseline
    print("\n[Model 1] Fitting Cross-Lingual TF-IDF Baseline...")
    df_aligned = loader.load_aligned_dataset()
    tfidf_model = CrossLingualTFIDFModel()
    tfidf_model.fit(
        en_corpus=list(df_aligned["english_original"]),
        hi_corpus=list(df_aligned["hindi_translation"])
    )
    print("  -> Cross-Lingual TF-IDF baseline fitted and saved successfully.")

    # 2. Fit Model 2: Multilingual Sentence Transformer Calibration
    print("\n[Model 2] Checking Sentence Transformer calibration data...")
    st_model = MultilingualSentenceTransformerModel()
    if len(df_annotated) >= 5:
        pairs = list(zip(df_annotated["english_original"], df_annotated["hindi_translation"]))
        scores = list(df_annotated["human_lmp_score"].astype(float))
        st_model.fit_calibration(pairs, scores)
        print("  -> Sentence Transformer calibrated on genuine human scores.")
    else:
        print("  -> Sentence Transformer: 0 genuine human English-Hindi annotations found.")
        print("     Operating in zero-shot cross-lingual cosine similarity mode.")

    # 3. Supervised Training Model 3: JUDGEBERT-X (XLM-RoBERTa)
    print("\n[Model 3] Checking JUDGEBERT-X (XLM-RoBERTa) training requirements...")
    if len(df_annotated) < 5:
        print("  -> [ZERO-FABRICATION RULE ENFORCED]")
        print("     Supervised training of XLM-RoBERTa regression head aborted.")
        print("     Reason: 0 genuine English-Hindi human annotations found.")
        print("     To train, populate data/english_hindi/annotated.csv with verified 1-10 human scores.")
        return {
            "status": "AWAITING_HUMAN_ANNOTATIONS",
            "tfidf_trained": True,
            "sentence_transformer_calibrated": False,
            "judgebert_xlmr_trained": False,
            "genuine_annotated_samples": 0,
            "message": "Model 1 fitted; Model 2 & Model 3 running in zero-shot cross-lingual mode pending human annotation."
        }

    # If genuine human scores exist, train the neural regression model
    train_df, val_df, _ = loader.load_splits()
    train_annotated = train_df.dropna(subset=["human_lmp_score"])
    val_annotated = val_df.dropna(subset=["human_lmp_score"])

    train_pairs = list(zip(train_annotated["english_original"], train_annotated["hindi_translation"]))
    train_scores = list(train_annotated["human_lmp_score"].astype(float))
    val_pairs = list(zip(val_annotated["english_original"], val_annotated["hindi_translation"]))
    val_scores = list(val_annotated["human_lmp_score"].astype(float))

    trainer = JudgeBertXXLMRTrainer(lr=2e-5, batch_size=8, seed=42)
    summary = trainer.fit(train_pairs, train_scores, val_pairs, val_scores, epochs=5, patience=2)

    return {
        "status": "SUPERVISED_TRAINING_COMPLETE",
        "tfidf_trained": True,
        "sentence_transformer_calibrated": True,
        "judgebert_xlmr_trained": True,
        "best_epoch": summary["best_epoch"],
        "best_val_loss": summary["best_val_loss"]
    }
