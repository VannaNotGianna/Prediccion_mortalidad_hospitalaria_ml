"""Definición de modelos y espacios de búsqueda de hiperparámetros (P2).

Cada modelo es un ``Pipeline(preprocessor -> classifier)``: el preprocesador
se reajusta en cada fold de validación cruzada, por lo que imputación,
codificación y escalamiento nunca ven datos de validación ni de prueba.

Modelos (diapositivas del curso):

- ``dummy``      : baseline de P1, ``DummyClassifier(strategy="prior")``.
- ``logreg``     : regresión logística regularizada.  p = σ(wᵀx + b);
                   minimiza log-loss + λΩ(w), con Ω = ||w||₂² (ridge) o
                   ||w||₁ (lasso).  En sklearn C = 1/λ.
- ``tree``       : árbol CART (Gini / entropía) con poda por complejidad de
                   costo  R_α(T) = R(T) + α|T|  (``ccp_alpha``).
- ``rf``         : Random Forest: bagging de árboles + submuestreo aleatorio
                   de variables en cada split; p̄(x) = (1/B) Σ_b P_b(y=1|x).
- ``gboost``     : Gradient boosting  F_t(x) = F_{t-1}(x) + η h_t(x)
                   (``HistGradientBoostingClassifier``, pérdida log-loss).
"""

from scipy.stats import loguniform, randint, uniform

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_preprocessor, get_feature_set


RANDOM_STATE = 42

# Nombres legibles para tablas y figuras.
MODEL_LABELS = {
    "dummy": "Dummy (prior)",
    "logreg": "Regresión logística",
    "tree": "Árbol de decisión",
    "rf": "Random Forest",
    "gboost": "Gradient Boosting",
}

# Prefijo común para los hiperparámetros del imputador numérico.
_INDICATOR = "preprocessor__numeric__imputer__add_indicator"


def _classifier(name):
    if name == "dummy":
        return DummyClassifier(strategy="prior")

    if name == "logreg":
        return LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )

    if name == "tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            # La paralelización se hace a nivel de búsqueda/CV.
            n_jobs=1,
            random_state=RANDOM_STATE,
        )

    if name == "gboost":
        return HistGradientBoostingClassifier(
            early_stopping=False,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Modelo desconocido: {name!r}")


def make_pipeline(name, feature_set="full", imputation="median"):
    """Pipeline sin ajustar para el modelo ``name`` y el conjunto de variables."""
    features = get_feature_set(feature_set)

    # Escalamiento solo para el modelo lineal regularizado: la penalización
    # λ||w|| depende de la escala de cada variable. Los árboles son invariantes
    # a transformaciones monótonas, por lo que no lo necesitan.
    scaled = name == "logreg"

    return Pipeline([
        ("preprocessor", build_preprocessor(
            scaled=scaled, features=features, imputation=imputation)),
        ("classifier", _classifier(name)),
    ])


def param_space(name):
    """Espacio de búsqueda y número de iteraciones de ``RandomizedSearchCV``.

    Devuelve ``(distribuciones, n_iter)``; ``n_iter=None`` indica que no hay
    hiperparámetros que ajustar.
    """
    if name == "dummy":
        return {}, None

    if name == "logreg":
        return {
            # C = 1/λ: valores pequeños = regularización fuerte.
            "classifier__C": loguniform(1e-3, 1e2),
            # 0 = ridge (L2), 1 = lasso (L1).
            "classifier__l1_ratio": [0.0, 1.0],
            "classifier__class_weight": [None, "balanced"],
            _INDICATOR: [False, True],
        }, 30

    if name == "tree":
        return {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [3, 4, 5, 6, 8, 10, None],
            "classifier__min_samples_leaf": randint(5, 100),
            "classifier__ccp_alpha": loguniform(1e-5, 1e-2),
            "classifier__class_weight": [None, "balanced"],
            _INDICATOR: [False, True],
        }, 40

    if name == "rf":
        return {
            "classifier__max_features": ["sqrt", "log2", 0.3, 0.5],
            "classifier__max_depth": [6, 10, 15, None],
            "classifier__min_samples_leaf": randint(1, 30),
            "classifier__class_weight": [None, "balanced", "balanced_subsample"],
            _INDICATOR: [False, True],
        }, 20

    if name == "gboost":
        return {
            "classifier__learning_rate": loguniform(0.01, 0.2),
            "classifier__max_iter": randint(100, 600),
            "classifier__max_leaf_nodes": randint(8, 48),
            "classifier__min_samples_leaf": randint(10, 100),
            "classifier__l2_regularization": uniform(0.0, 2.0),
            "classifier__max_features": [0.5, 0.8, 1.0],
            _INDICATOR: [False, True],
        }, 30

    raise ValueError(f"Modelo desconocido: {name!r}")
