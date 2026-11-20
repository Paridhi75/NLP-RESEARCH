"""
Data Loader Module for English-to-Hindi Legal Meaning Preservation (LMP).
Loads aligned datasets, leakage-safe splits, and verified human annotation files.
"""
import os
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import GroupShuffleSplit

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
EN_HI_DIR = os.path.join(DATA_DIR, "english_hindi")
SPLITS_DIR = os.path.join(EN_HI_DIR, "splits")


class EnglishHindiDataLoader:
    """Manages English-to-Hindi aligned legal datasets and partitions."""

    def __init__(self, data_dir: str = EN_HI_DIR):
        self.data_dir = data_dir
        self.aligned_path = os.path.join(self.data_dir, "aligned_dataset.csv")
        self.annotated_path = os.path.join(self.data_dir, "annotated.csv")
        self.template_path = os.path.join(self.data_dir, "annotation_template.csv")
        self.splits_dir = os.path.join(self.data_dir, "splits")

    def load_aligned_dataset(self) -> pd.DataFrame:
        """Loads all aligned English-Hindi pairs."""
        if not os.path.exists(self.aligned_path):
            raise FileNotFoundError(f"Aligned dataset missing at {self.aligned_path}. Run data preparation first.")
        return pd.read_csv(self.aligned_path)

    def load_annotated_dataset(self) -> pd.DataFrame:
        """Loads genuine human-annotated English-Hindi pairs."""
        if not os.path.exists(self.annotated_path):
            return pd.DataFrame(columns=["english_original", "hindi_translation", "human_lmp_score", "legal_error_category", "domain"])
        df = pd.read_csv(self.annotated_path)
        # Filter strictly for valid numerical scores
        if "human_lmp_score" in df.columns:
            return df[pd.to_numeric(df["human_lmp_score"], errors="coerce").notnull()].copy()
        return pd.DataFrame(columns=df.columns)

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads train, validation, and test splits."""
        train_p = os.path.join(self.splits_dir, "train.csv")
        val_p = os.path.join(self.splits_dir, "val.csv")
        test_p = os.path.join(self.splits_dir, "test.csv")

        if not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
            self.create_splits()

        return pd.read_csv(train_p), pd.read_csv(val_p), pd.read_csv(test_p)

    def create_splits(self, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
        """Creates zero-leakage clause-grouped train/val/test splits."""
        df = self.load_aligned_dataset()
        os.makedirs(self.splits_dir, exist_ok=True)

        # Extract base clause identifier (e.g. INS01 from INS01-ENHI-F)
        df["base_clause"] = df["pair_id"].apply(lambda x: str(x).split("-")[0])

        gss1 = GroupShuffleSplit(n_splits=1, test_size=(test_size + val_size), random_state=seed)
        train_idx, temp_idx = next(gss1.split(df, groups=df["base_clause"]))

        df_train = df.iloc[train_idx].copy()
        df_temp = df.iloc[temp_idx].copy()

        rel_test_size = test_size / (test_size + val_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test_size, random_state=seed)
        val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["base_clause"]))

        df_val = df_temp.iloc[val_idx].copy()
        df_test = df_temp.iloc[test_idx].copy()

        df_train.drop(columns=["base_clause"]).to_csv(os.path.join(self.splits_dir, "train.csv"), index=False)
        df_val.drop(columns=["base_clause"]).to_csv(os.path.join(self.splits_dir, "val.csv"), index=False)
        df_test.drop(columns=["base_clause"]).to_csv(os.path.join(self.splits_dir, "test.csv"), index=False)

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates accurate dataset statistics without fabricating annotations."""
        df_aligned = self.load_aligned_dataset()
        df_annotated = self.load_annotated_dataset()

        return {
            "total_aligned_pairs": len(df_aligned),
            "insurance_pairs": int((df_aligned["domain"] == "Insurance").sum()),
            "tenancy_pairs": int((df_aligned["domain"] == "Tenancy").sum()),
            "faithful_translations": int((df_aligned["translation_type"] == "faithful").sum()),
            "perturbed_translations": int((df_aligned["translation_type"] == "perturbed").sum()),
            "genuine_human_annotated_pairs": len(df_annotated),
            "pending_annotation_pairs": len(df_aligned) - len(df_annotated),
            "annotation_status": "PENDING_EXPERT_HUMAN_ANNOTATION" if len(df_annotated) == 0 else "PARTIALLY_ANNOTATED"
        }
