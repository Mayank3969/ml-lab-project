import sys
from pathlib import Path

# Make sure we can import from utils/
MODELS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODELS_DIR))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from utils.config import PROCESSED, STAGE1_FEATURES, STAGE1_NUM, STAGE1_CAT, STAGE1_TARGET


def load_merged():
    """Load the merged train/test CSVs with collision_index as text."""
    train = pd.read_csv(PROCESSED / "merged_train.csv",
                        converters={"collision_index": str})
    test = pd.read_csv(PROCESSED / "merged_test.csv",
                       converters={"collision_index": str})
    return train, test


def collapse_to_collision(df):
    """One row per collision (Stage 1 operates at collision level)."""
    keep = ["collision_index", STAGE1_TARGET] + STAGE1_FEATURES
    return df[keep].drop_duplicates(subset=["collision_index"]).reset_index(drop=True)


def build_stage1_preprocessor():
    """ColumnTransformer: passthrough for speed_limit, one-hot for categoricals."""
    return ColumnTransformer([
        ("num", "passthrough", STAGE1_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), STAGE1_CAT),
    ])