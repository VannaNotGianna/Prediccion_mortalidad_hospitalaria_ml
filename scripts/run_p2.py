"""Pipeline reproducible de P2: selección de modelos, evaluación y análisis.

Uso (desde la raíz del repositorio):

    python scripts/run_p2.py            # experimento completo
    python scripts/run_p2.py --quick    # versión reducida para pruebas rápidas

Protocolo
---------
1. Particiones fijas de P1 (80/20 estratificado, semilla 42).
2. Selección de hiperparámetros con ``RandomizedSearchCV`` + validación
   cruzada estratificada de 5 folds, SOLO sobre entrenamiento. Métrica de
   selección: ROC-AUC (independiente del umbral y robusta al desbalance).
3. Umbral de decisión τ elegido sobre predicciones out-of-fold de
   entrenamiento (maximiza F1). Se reportan también resultados con τ = 0.5.
4. Modelo final = mayor ROC-AUC media en CV (conjunto de variables completo).
5. El conjunto de prueba se usa una única vez, al final, para reportar
   métricas; no interviene en ninguna decisión.
6. Ablación por conjuntos de variables (demográficas vs. fisiológicas)
   para responder la pregunta de investigación.

Salidas: ``results/tables/p2_*.csv``, ``results/figures/p2_*.png``,
``results/p2_run_metadata.json`` y modelos en ``results/models/`` (ignorado
por git).
"""

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    validation_curve,
)

ROOT = Path(__file__).resolve().parents[1]   # raíz del repositorio
sys.path.insert(0, str(ROOT))

from preprocessing import FEATURE_GROUPS, get_feature_set, load_data  # noqa: E402
from src.evaluation import (  # noqa: E402
    MODEL_COLORS,
    bootstrap_ci,
    classification_metrics,
    plot_barh,
    plot_calibration,
    plot_confusion_matrices,
    plot_roc_pr,
    plot_validation_curve,
    select_threshold,
    subgroup_table,
)
from src.models import (  # noqa: E402
    MODEL_LABELS,
    RANDOM_STATE,
    make_pipeline,
    param_space,
)

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"

SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
    "neg_log_loss": "neg_log_loss",
}

FULL_MODELS = ["dummy", "logreg", "tree", "rf", "gboost"]
ABLATION_MODELS = ["logreg", "gboost"]
ABLATION_SETS = ["demographic", "demographic+physiological"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tune(name, feature_set, X, y, cv, quick=False):
    """Búsqueda de hiperparámetros por CV estratificada (solo train)."""
    pipe = make_pipeline(name, feature_set)
    space, n_iter = param_space(name)

    if n_iter is None:
        space, n_iter = {"classifier__strategy": ["prior"]}, 1
    if quick:
        n_iter = min(n_iter, 3)

    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=n_iter,
        scoring=SCORING,
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=True,
        error_score="raise",
    )
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        search.fit(X, y)
    elapsed = time.time() - start

    res = search.cv_results_
    i = search.best_index_
    row = {
        "model": name,
        "feature_set": feature_set,
        "n_candidates": n_iter,
        "fit_seconds": round(elapsed, 1),
        "best_params": json.dumps(
            {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                 else str(v))
             for k, v in search.best_params_.items()},
            default=str,
        ),
        "train_roc_auc": res["mean_train_roc_auc"][i],
    }
    for metric in SCORING:
        row[f"cv_{metric}_mean"] = res[f"mean_test_{metric}"][i]
        row[f"cv_{metric}_std"] = res[f"std_test_{metric}"][i]

    # Predicciones out-of-fold en entrenamiento → umbral τ (sin tocar test).
    oof = cross_val_predict(
        clone(search.best_estimator_), X, y, cv=cv,
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    if name == "dummy":
        tau, f1_tau = 0.5, 0.0   # el baseline de P1 se reporta con τ = 0.5
    else:
        tau, f1_tau = select_threshold(y, oof)
    row["threshold"] = tau
    row["oof_f1_at_threshold"] = f1_tau
    row["oof_roc_auc"] = classification_metrics(y, oof)["roc_auc"]

    log(f"{name:7s} | {feature_set:26s} | CV AUC "
        f"{row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f} "
        f"| τ={tau:.3f} | {elapsed:.0f}s")
    return search.best_estimator_, row


def split_hash():
    h = hashlib.sha256()
    for f in ("train_indices.csv", "test_indices.csv"):
        h.update((ROOT / "dataset" / "processed" / f).read_bytes())
    return h.hexdigest()[:16]


def main(quick=False):
    global TABLES, FIGURES, MODELS
    if quick:
        # El modo rápido nunca sobrescribe los resultados oficiales.
        base = ROOT / "results" / "quick"
        TABLES, FIGURES, MODELS = base / "tables", base / "figures", base / "models"

    for folder in (TABLES, FIGURES, MODELS):
        folder.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    assert not set(X_train.index) & set(X_test.index), "Particiones solapadas"
    log(f"Train {X_train.shape}, test {X_test.shape}, "
        f"prevalencia train={y_train.mean():.4f} test={y_test.mean():.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. Selección de hiperparámetros (solo entrenamiento)
    # ------------------------------------------------------------------
    fitted, rows = {}, []
    for name in FULL_MODELS:
        est, row = tune(name, "full", X_train, y_train, cv, quick)
        fitted[(name, "full")] = est
        rows.append(row)
    for name in ABLATION_MODELS:
        for fset in ABLATION_SETS:
            est, row = tune(name, fset, X_train, y_train, cv, quick)
            fitted[(name, fset)] = est
            rows.append(row)

    cv_table = pd.DataFrame(rows)
    cv_table.to_csv(TABLES / "p2_cv_results.csv", index=False)

    # Modelo final: mayor ROC-AUC en CV entre los modelos con todas las variables.
    candidates = cv_table[(cv_table["feature_set"] == "full")
                          & (cv_table["model"] != "dummy")]
    best_name = candidates.sort_values(
        "cv_roc_auc_mean", ascending=False).iloc[0]["model"]
    log(f"Modelo seleccionado por CV: {best_name}")

    thresholds = {(r["model"], r["feature_set"]): r["threshold"] for r in rows}

    # ------------------------------------------------------------------
    # 2. Evaluación final en prueba (una sola vez)
    # ------------------------------------------------------------------
    test_rows, probas = [], {}
    for (name, fset), est in fitted.items():
        proba = est.predict_proba(X_test)[:, 1]
        probas[(name, fset)] = proba
        for kind, tau in (("0.5", 0.5), ("cv_f1", thresholds[(name, fset)])):
            if name == "dummy" and kind == "cv_f1":
                continue
            m = classification_metrics(y_test, proba, tau)
            test_rows.append({"model": name, "feature_set": fset,
                              "threshold_rule": kind, **m})
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(TABLES / "p2_test_metrics.csv", index=False)

    # Comprobación de consistencia con el baseline de P1.
    dummy_auc = test_table.query("model == 'dummy'")["roc_auc"].iloc[0]
    assert abs(dummy_auc - 0.5) < 1e-9

    pred_df = pd.DataFrame({"original_index": X_test.index,
                            "y_true": y_test.values})
    for (name, fset), p in probas.items():
        pred_df[f"proba_{name}_{fset}"] = p
    pred_df.to_csv(TABLES / "p2_test_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Intervalos bootstrap y comparaciones pareadas
    # ------------------------------------------------------------------
    keyed = {f"{n}|{f}": p for (n, f), p in probas.items() if n != "dummy"}
    boot_best = bootstrap_ci(y_test, keyed, reference=f"{best_name}|full",
                             n_boot=200 if quick else 1000)
    boot_best.to_csv(TABLES / "p2_bootstrap_vs_best.csv", index=False)
    boot_demo = bootstrap_ci(y_test, keyed, reference="logreg|demographic",
                             n_boot=200 if quick else 1000)
    boot_demo.to_csv(TABLES / "p2_bootstrap_vs_demographic.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Figuras de comparación
    # ------------------------------------------------------------------
    full_probas = {n: probas[(n, "full")] for n in FULL_MODELS if n != "dummy"}
    full_probas["logreg_demographic"] = probas[("logreg", "demographic")]
    labels = dict(MODEL_LABELS)
    labels["logreg_demographic"] = "Reg. logística (solo demográficas)"
    plot_roc_pr(y_test, full_probas, labels, y_test.mean(),
                FIGURES / "p2_roc_pr_curves.png")
    plot_calibration(y_test, {k: full_probas[k] for k in
                              ("logreg", "rf", "gboost", "logreg_demographic")},
                     labels, FIGURES / "p2_calibration.png")

    matrices = {}
    for n in FULL_MODELS:
        tau = 0.5 if n == "dummy" else thresholds[(n, "full")]
        m = classification_metrics(y_test, probas[(n, "full")], tau)
        matrices[f"{MODEL_LABELS[n]}\n(τ={tau:.2f})"] = np.array(
            [[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    plot_confusion_matrices(matrices, FIGURES / "p2_confusion_matrices.png")

    # Curva de validación del árbol (sesgo-varianza en función de max_depth).
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    tree_pipe = make_pipeline("tree", "full").set_params(
        classifier__ccp_alpha=0.0, classifier__min_samples_leaf=1)
    tr_s, va_s = validation_curve(
        tree_pipe, X_train, y_train, param_name="classifier__max_depth",
        param_range=depths, cv=cv, scoring="roc_auc", n_jobs=-1)
    pd.DataFrame({"max_depth": depths,
                  "train_auc_mean": tr_s.mean(1), "train_auc_std": tr_s.std(1),
                  "cv_auc_mean": va_s.mean(1), "cv_auc_std": va_s.std(1)}
                 ).to_csv(TABLES / "p2_tree_validation_curve.csv", index=False)
    plot_validation_curve(depths, tr_s, va_s, "max_depth (sin poda)",
                          "Árbol de decisión: curva de validación",
                          FIGURES / "p2_tree_validation_curve.png")

    # ------------------------------------------------------------------
    # 5. Interpretación
    # ------------------------------------------------------------------
    lr = fitted[("logreg", "full")]
    names = lr.named_steps["preprocessor"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    coefs = lr.named_steps["classifier"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": names, "coef": coefs,
                            "odds_ratio": np.exp(coefs)})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    coef_df.to_csv(TABLES / "p2_logreg_coefficients.csv", index=False)
    top = coef_df.head(20).set_index("feature")["coef"]
    plot_barh(top, None,
              "Regresión logística: 20 coeficientes de mayor magnitud",
              "Coeficiente (log-odds por 1 DE o por categoría)",
              FIGURES / "p2_logreg_coefficients.png",
              color=MODEL_COLORS["logreg"], reference_line=0)

    # Importancia por permutación del modelo final, sobre variables originales.
    # Se calcula en prueba solo con fines interpretativos (no se usa para
    # seleccionar variables ni modelos).
    best = fitted[(best_name, "full")]
    perm = permutation_importance(
        best, X_test, y_test, scoring="roc_auc",
        n_repeats=3 if quick else 20, random_state=RANDOM_STATE, n_jobs=-1)
    group_of = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "group": [group_of[c] for c in X_test.columns],
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(TABLES / "p2_permutation_importance.csv", index=False)
    plot_barh(perm_df.set_index("feature")["importance_mean"],
              perm_df.set_index("feature")["importance_std"],
              f"Importancia por permutación — {MODEL_LABELS[best_name]}",
              "Disminución media de ROC-AUC en prueba",
              FIGURES / "p2_permutation_importance.png",
              color=MODEL_COLORS[best_name], reference_line=0)

    # ------------------------------------------------------------------
    # 6. Análisis de errores y subgrupos (modelo final, umbral τ de CV)
    # ------------------------------------------------------------------
    tau_best = thresholds[(best_name, "full")]
    p_best = probas[(best_name, "full")]
    sub = pd.concat([
        subgroup_table(X_test, y_test, p_best, tau_best, "dzgroup"),
        subgroup_table(X_test, y_test, p_best, tau_best, "sex"),
        subgroup_table(X_test, y_test, p_best, tau_best, "race"),
        subgroup_table(X_test, y_test, p_best, tau_best, "income"),
        subgroup_table(X_test, y_test, p_best, tau_best, "age",
                       bins=[0, 50, 65, 80, 200],
                       labels=["<50", "50-64", "65-79", ">=80"]),
    ])
    sub.to_csv(TABLES / "p2_subgroup_performance.csv", index=False)

    pred = (p_best >= tau_best).astype(int)
    outcome = np.select(
        [(y_test == 1) & (pred == 1), (y_test == 1) & (pred == 0),
         (y_test == 0) & (pred == 1)],
        ["TP", "FN", "FP"], default="TN")
    numeric = X_test.select_dtypes("number").columns
    profile = X_test[numeric].assign(outcome=outcome).groupby("outcome").median().T
    profile["missing_pct_FN"] = X_test.loc[outcome == "FN", numeric].isna().mean() * 100
    profile["missing_pct_TP"] = X_test.loc[outcome == "TP", numeric].isna().mean() * 100
    profile = profile[["TP", "FN", "FP", "TN", "missing_pct_TP", "missing_pct_FN"]]
    profile.to_csv(TABLES / "p2_error_profile.csv")
    counts = pd.Series(outcome).value_counts()
    log(f"Errores del modelo final: {counts.to_dict()}")

    # ------------------------------------------------------------------
    # 7. Persistencia y metadatos
    # ------------------------------------------------------------------
    for (name, fset), est in fitted.items():
        joblib.dump(est, MODELS / f"{name}__{fset.replace('+', '_')}.joblib")

    meta = {
        "quick": quick,
        "random_state": RANDOM_STATE,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "selection_metric": "roc_auc",
        "threshold_rule": "max F1 on out-of-fold train predictions",
        "selected_model": best_name,
        "selected_threshold": tau_best,
        "split_sha256_16": split_hash(),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "feature_sets": {k: get_feature_set(k) for k in
                         ["demographic", "demographic+physiological", "full"]},
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (TABLES.parent / "p2_run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = test_table.query("feature_set == 'full'")[
        ["model", "threshold_rule", "threshold", "accuracy", "precision",
         "recall", "f1", "roc_auc", "average_precision"]]
    print("\n=== Prueba (todas las variables) ===")
    print(summary.round(4).to_string(index=False))
    log("P2 completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true",
                        help="Búsqueda reducida (3 candidatos) para pruebas.")
    main(**vars(parser.parse_args()))
