"""Pruebas de integridad del pipeline de P2.

Ejecutar desde la raíz:  python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing import (  # noqa: E402
    FEATURE_GROUPS, build_preprocessor, get_feature_set, load_config, load_data,
)
from src.evaluation import classification_metrics, select_threshold  # noqa: E402
from src.models import make_pipeline, param_space  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_split_sizes_and_disjoint(data):
    X_train, X_test, y_train, y_test = data
    assert X_train.shape == (7284, 27)
    assert X_test.shape == (1821, 27)
    assert not set(X_train.index) & set(X_test.index)
    # Estratificación: misma prevalencia (±0.5 p.p.)
    assert abs(y_train.mean() - y_test.mean()) < 0.005


def test_no_leakage_columns_in_features():
    config = load_config()
    leaky = set(config["initial_excluded_columns"]) | {config["target"]}
    assert not leaky & set(config["selected_features"])


def test_feature_groups_partition_selected_features():
    groups = list(FEATURE_GROUPS.values())
    union = set().union(*groups)
    assert union == set(load_config()["selected_features"])
    assert sum(len(g) for g in groups) == len(union)  # disjuntos
    assert get_feature_set("full") == load_config()["selected_features"]
    assert set(get_feature_set("demographic")) == set(FEATURE_GROUPS["demographic"])


def test_default_preprocessor_backward_compatible(data):
    X_train, *_ = data
    pre = build_preprocessor(scaled=True).fit(X_train)
    out = pre.transform(X_train)
    n_cat = sum(X_train[c].nunique() for c in load_config()["categorical_features"])
    assert out.shape == (len(X_train), 21 + n_cat)
    assert not np.isnan(np.asarray(out, dtype=float)).any()


def test_imputer_learns_only_from_train(data):
    X_train, X_test, _, _ = data
    pre = build_preprocessor().fit(X_train)
    imputer = pre.named_transformers_["numeric"].named_steps["imputer"]
    numeric = load_config()["numeric_features"]
    np.testing.assert_allclose(imputer.statistics_, X_train[numeric].median().values)
    full_median = pd.concat([X_train, X_test])[numeric].median().values
    assert not np.allclose(imputer.statistics_, full_median)


@pytest.mark.parametrize("strategy", ["median", "normal", "knn", "iterative"])
@pytest.mark.parametrize("indicator", [False, True])
def test_imputation_strategies_leave_no_nan(data, strategy, indicator):
    X_train, X_test, _, _ = data
    pre = build_preprocessor(scaled=True, imputation=strategy,
                             add_indicator=indicator).fit(X_train.iloc[:1500])
    out = np.asarray(pre.transform(X_test), dtype=float)
    assert not np.isnan(out).any()
    assert len(pre.get_feature_names_out()) == out.shape[1]


def test_normal_value_imputer_uses_support_values(data):
    X_train, *_ = data
    from preprocessing import NormalValueImputer, SUPPORT_NORMAL_VALUES
    cols = ["alb", "bili", "glucose"]
    imp = NormalValueImputer().fit(X_train[cols])
    assert imp.fill_values_[0] == SUPPORT_NORMAL_VALUES["alb"]
    assert imp.fill_values_[1] == SUPPORT_NORMAL_VALUES["bili"]
    # Sin valor normal definido → mediana de entrenamiento.
    assert imp.fill_values_[2] == X_train["glucose"].median()


def test_subset_preprocessor_uses_only_subset(data):
    X_train, *_ = data
    feats = get_feature_set("demographic")
    pre = build_preprocessor(features=feats).fit(X_train)
    # El remainder aparece en transformers_ con transformer="drop".
    used = {c for _, trans, cols in pre.transformers_ if trans != "drop"
            for c in cols}
    assert used == set(feats)
    assert len(pre.get_feature_names_out()) == pre.transform(X_train).shape[1]


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    m = classification_metrics(y, p, 0.5)
    yhat = (p >= 0.5).astype(int)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, yhat))
    assert m["precision"] == pytest.approx(precision_score(y, yhat))
    assert m["recall"] == pytest.approx(recall_score(y, yhat))
    assert m["f1"] == pytest.approx(f1_score(y, yhat))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["specificity"] == pytest.approx(m["TN"] / (m["TN"] + m["FP"]))
    tau, f1 = select_threshold(y, p)
    assert classification_metrics(y, p, tau)["f1"] == pytest.approx(f1)


def test_dummy_reproduces_p1_baseline(data):
    X_train, X_test, y_train, y_test = data
    model = make_pipeline("dummy").fit(X_train, y_train)
    m = classification_metrics(y_test, model.predict_proba(X_test)[:, 1])
    ref = pd.read_csv(ROOT / "results" / "tables" / "baseline_metrics.csv").iloc[0]
    assert m["accuracy"] == pytest.approx(ref["Accuracy"])
    assert m["roc_auc"] == pytest.approx(ref["ROC-AUC"])
    assert m["average_precision"] == pytest.approx(ref["Average Precision"])


@pytest.mark.parametrize("name", ["logreg", "tree", "rf", "gboost"])
def test_every_model_fits_with_search_params(data, name):
    X_train, X_test, y_train, _ = data
    space, _ = param_space(name)
    pipe = make_pipeline(name)
    # Primer valor de cada lista / muestra de cada distribución.
    params = {k: (v[-1] if isinstance(v, list) else v.rvs(random_state=0))
              for k, v in space.items()}
    pipe.set_params(**params)
    pipe.fit(X_train.iloc[:800], y_train.iloc[:800])
    proba = pipe.predict_proba(X_test.iloc[:50])[:, 1]
    assert proba.shape == (50,) and ((proba >= 0) & (proba <= 1)).all()
