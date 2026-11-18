"""Carga de particiones y construcción del preprocesador (P1 → P2).

Este módulo es la única fuente de verdad para:

- las particiones train/test fijadas en P1 (``dataset/processed/*_indices.csv``);
- la lista de 27 variables seleccionadas (``preprocessing_config.json``);
- el preprocesador *sin ajustar* que debe colocarse dentro de un ``Pipeline``
  para que imputación, codificación y escalamiento se aprendan solo con los
  datos de entrenamiento de cada fold (prevención de data leakage).
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "dataset" / "processed"


# ---------------------------------------------------------------------------
# Grupos de variables (P2).
# Se usan para responder la pregunta de investigación: ¿cuánto aportan las
# variables fisiológicas frente a las puramente demográficas?
# La unión de los tres grupos es exactamente ``selected_features`` del config.
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    # Características del paciente, independientes de su estado clínico.
    "demographic": ["age", "sex", "race", "income", "edu"],
    # Diagnóstico, comorbilidades y estado funcional previo / al ingreso.
    "clinical": [
        "dzgroup", "dzclass", "num.co", "diabetes", "dementia",
        "ca", "adls", "hday",
    ],
    # Mediciones fisiológicas y de laboratorio (día 3 del estudio).
    "physiological": [
        "scoma", "meanbp", "wblc", "hrt", "resp", "temp", "pafi",
        "alb", "bili", "crea", "sod", "ph", "glucose", "bun",
    ],
}

# Conjuntos de variables evaluados en P2 (combinaciones de grupos).
FEATURE_SETS = {
    "demographic": ["demographic"],
    "demographic+physiological": ["demographic", "physiological"],
    "full": ["demographic", "clinical", "physiological"],
}


# Valores "normales" recomendados por la documentación oficial de SUPPORT
# (https://hbiostat.org/data/repo/supportdesc, "normal fill-in values"):
# un laboratorio no solicitado suele indicar que el médico lo esperaba normal.
SUPPORT_NORMAL_VALUES = {
    "alb": 3.5,
    "pafi": 333.3,
    "bili": 1.01,
    "crea": 1.01,
    "bun": 6.51,
    "wblc": 9.0,
}

IMPUTATION_STRATEGIES = ("median", "normal", "knn", "iterative")


class NormalValueImputer(BaseEstimator, TransformerMixin):
    """Imputación con valores normales de SUPPORT; mediana (de train) para el resto.

    Ajusta solo las medianas de las columnas sin valor normal definido, por
    lo que también se aprende exclusivamente con datos de entrenamiento.
    """

    def __init__(self, normal_values=None, add_indicator=False):
        self.normal_values = normal_values
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        normals = (SUPPORT_NORMAL_VALUES if self.normal_values is None
                   else self.normal_values)
        medians = X.median()
        self.fill_values_ = np.array([
            normals.get(col, medians[col]) for col in X.columns
        ], dtype=float)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.indicator_cols_ = np.where(X.isna().any().values)[0]
        return self

    def transform(self, X):
        values = pd.DataFrame(X).to_numpy(dtype=float, copy=True)
        mask = np.isnan(values)
        values[mask] = np.take(self.fill_values_, np.where(mask)[1])
        if self.add_indicator:
            values = np.hstack(
                [values, mask[:, self.indicator_cols_].astype(float)])
        return values

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.add_indicator:
            names += [f"missingindicator_{self.feature_names_in_[i]}"
                      for i in self.indicator_cols_]
        return np.asarray(names, dtype=object)


def _numeric_imputer(strategy, add_indicator):
    if strategy == "median":
        return SimpleImputer(strategy="median", add_indicator=add_indicator)
    if strategy == "normal":
        return NormalValueImputer(add_indicator=add_indicator)
    if strategy == "knn":
        return KNNImputer(n_neighbors=10, add_indicator=add_indicator)
    if strategy == "iterative":
        return IterativeImputer(max_iter=10, random_state=42,
                                add_indicator=add_indicator)
    raise ValueError(
        f"Imputación desconocida: {strategy!r}. Opciones: {IMPUTATION_STRATEGIES}")


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


def get_feature_set(name="full"):
    """Devolver la lista de columnas de un conjunto de ``FEATURE_SETS``.

    Se respeta el orden de ``selected_features`` del config.
    """
    if name not in FEATURE_SETS:
        raise ValueError(
            f"Conjunto desconocido: {name!r}. "
            f"Opciones: {list(FEATURE_SETS)}"
        )

    wanted = {
        col
        for group in FEATURE_SETS[name]
        for col in FEATURE_GROUPS[group]
    }

    return [
        col for col in load_config()["selected_features"]
        if col in wanted
    ]


def build_preprocessor(scaled=False, features=None, add_indicator=False,
                       imputation="median"):
    """Construir un preprocesador nuevo, todavía sin ajustar.

    Parameters
    ----------
    scaled : bool
        Añadir ``StandardScaler`` a las variables numéricas
        (regresión logística). Árboles y ensambles no lo requieren.
    features : list[str] | None
        Subconjunto de columnas a utilizar. ``None`` usa las 27 variables
        seleccionadas en P1. Las columnas no incluidas se descartan
        (``remainder="drop"``).
    add_indicator : bool
        Añadir indicadores binarios de valor faltante a las variables
        numéricas imputadas. Es un hiperparámetro que se ajusta por CV.
    imputation : {"median", "normal", "knn", "iterative"}
        Estrategia para variables numéricas. ``"normal"`` usa los valores
        normales de SUPPORT; ``"knn"`` (k = 10) e ``"iterative"`` (tipo MICE)
        se aplican tras estandarizar, porque dependen de distancias /
        regresiones entre variables. Todas se ajustan solo con train.
    """
    config = load_config()

    numeric = config["numeric_features"]
    categorical = config["categorical_features"]

    if features is not None:
        unknown = set(features) - set(numeric) - set(categorical)
        if unknown:
            raise ValueError(f"Variables no reconocidas: {sorted(unknown)}")

        numeric = [col for col in numeric if col in features]
        categorical = [col for col in categorical if col in features]

    imputer = _numeric_imputer(imputation, add_indicator)

    if imputation in ("knn", "iterative"):
        # StandardScaler ignora NaN al ajustar y los conserva al transformar.
        numeric_steps = [("scaler", StandardScaler()), ("imputer", imputer)]
    else:
        numeric_steps = [("imputer", imputer)]
        if scaled:
            numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric:
        transformers.append(("numeric", numeric_pipeline, numeric))

    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
