
from pathlib import Path
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "dataset" / "processed"


def load_config():
    """Cargar la configuración de preprocesamiento."""
    config_path = PROCESSED_DIR / "preprocessing_config.json"

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_data():
    """Recuperar exactamente las particiones de P2."""
    config = load_config()

    df = pd.read_csv(ROOT / "dataset" / "raw" / "support2.csv")

    train_idx = pd.read_csv(
        PROCESSED_DIR / "train_indices.csv"
    )["original_index"].tolist()

    test_idx = pd.read_csv(
        PROCESSED_DIR / "test_indices.csv"
    )["original_index"].tolist()

    features = config["selected_features"]
    target = config["target"]

    X_train = df.loc[train_idx, features].copy()
    X_test = df.loc[test_idx, features].copy()

    y_train = df.loc[train_idx, target].astype(int).copy()
    y_test = df.loc[test_idx, target].astype(int).copy()

    return X_train, X_test, y_train, y_test


def build_preprocessor(scaled=False):
    """Construir un preprocesador nuevo, todavía sin ajustar."""
    config = load_config()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scaled:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                config["numeric_features"]
            ),
            (
                "categorical",
                categorical_pipeline,
                config["categorical_features"]
            )
        ],
        remainder="drop"
    )
