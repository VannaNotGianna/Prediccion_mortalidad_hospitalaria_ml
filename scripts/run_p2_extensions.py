"""Extensiones de P2: imputación, recalibración y umbrales por costo / grupo.

Uso (desde la raíz, después de ``python scripts/run_p2.py``):

    python scripts/run_p2_extensions.py

Parte de los modelos ya ajustados por ``run_p2.py`` (hiperparámetros elegidos
por CV) y estudia, siempre con decisiones tomadas SOLO en entrenamiento:

A. Estrategias de imputación de variables numéricas: mediana (P1), valores
   normales de SUPPORT, KNN (k = 10) e iterativa tipo MICE. Comparación por CV
   estratificada de 5 folds (mismos folds para todas → comparación pareada).
B. Recalibración de los modelos con ``class_weight`` (regresión logística y
   Random Forest): Platt (sigmoide) e isotónica con ``CalibratedClassifierCV``.
   La elección del método se hace con CV anidada sobre train.
C. Umbral con costos clínicos explícitos: C_FP = 1, C_FN = R. Con
   probabilidades calibradas, la regla de Bayes da τ* = 1 / (1 + R); además se
   busca el τ empírico que minimiza FP + R·FN en predicciones OOF de train.
   Como no existe un R clínicamente validado para este problema, se presenta
   un análisis de sensibilidad R ∈ {1, 2, 3, 5, 10}.
D. Umbrales por grupo diagnóstico (``dzgroup``) frente a un umbral global.

El conjunto de prueba solo se usa para reportar el resultado de cada decisión.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_validate,
)

ROOT = Path(__file__).resolve().parents[1]   # raíz del repositorio
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))      # para importar run_p2

from preprocessing import IMPUTATION_STRATEGIES, load_data  # noqa: E402
from src.evaluation import (  # noqa: E402
    INK, INK_2, MODEL_COLORS, SURFACE, _style_axes,
    classification_metrics, expected_calibration_error, expected_cost,
    select_cost_threshold, select_threshold,
)
from src.models import MODEL_LABELS, RANDOM_STATE, make_pipeline  # noqa: E402

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"

COST_RATIOS = [1, 2, 3, 5, 10]
GROUP_COST_RATIO = 3          # R usado para comparar umbral global vs. por grupo
IMPUTATION_LABELS = {
    "median": "Mediana (P1)",
    "normal": "Valores normales SUPPORT",
    "knn": "KNN (k=10)",
    "iterative": "Iterativa (MICE)",
}
# Mejora mínima de AUC en CV para abandonar la imputación por mediana.
MIN_AUC_GAIN = 0.005


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tuned(name):
    path = MODELS / f"{name}__full.joblib"
    if not path.exists():
        log("Modelos no encontrados: ejecutando run_p2.py primero.")
        import run_p2
        run_p2.main()
    return joblib.load(path)


def with_imputation(tuned, name, strategy):
    """Mismo clasificador e indicador de faltantes, otra imputación."""
    indicator = (tuned.named_steps["preprocessor"]
                 .named_transformers_["numeric"]
                 .named_steps["imputer"].add_indicator)
    pipe = make_pipeline(name, "full", imputation=strategy)
    pipe.set_params(
        classifier=clone(tuned.named_steps["classifier"]),
        preprocessor__numeric__imputer__add_indicator=indicator,
    )
    return pipe


def score_row(y, p, **extra):
    m = classification_metrics(y, p)
    return {**extra, "roc_auc": m["roc_auc"],
            "average_precision": m["average_precision"],
            "brier": m["brier"], "log_loss": m["log_loss"],
            "ece": expected_calibration_error(y, p)}


def main():
    X_train, X_test, y_train, y_test = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    meta = json.loads((ROOT / "results" / "p2_run_metadata.json")
                      .read_text(encoding="utf-8"))
    best_name = meta["selected_model"]
    models = ["logreg", "tree", "rf", "gboost"]
    tuned = {n: load_tuned(n) for n in models}

    # ------------------------------------------------------------------
    # A. Imputación
    # ------------------------------------------------------------------
    rows, fold_auc = [], {}
    for name in models:
        for strategy in IMPUTATION_STRATEGIES:
            pipe = with_imputation(tuned[name], name, strategy)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cross_validate(
                    pipe, X_train, y_train, cv=cv, n_jobs=-1,
                    scoring={"roc_auc": "roc_auc",
                             "average_precision": "average_precision",
                             "brier": "neg_brier_score"})
            fold_auc[(name, strategy)] = res["test_roc_auc"]
            rows.append({
                "model": name, "imputation": strategy,
                "cv_roc_auc_mean": res["test_roc_auc"].mean(),
                "cv_roc_auc_std": res["test_roc_auc"].std(),
                "cv_average_precision_mean": res["test_average_precision"].mean(),
                "cv_brier_mean": -res["test_brier"].mean(),
            })
            log(f"Imputación {name:7s} {strategy:9s} CV AUC "
                f"{res['test_roc_auc'].mean():.4f}")
    imp = pd.DataFrame(rows)
    # Diferencia pareada por fold frente a la mediana.
    imp["cv_auc_diff_vs_median"] = [
        (fold_auc[(r.model, r.imputation)]
         - fold_auc[(r.model, "median")]).mean() for r in imp.itertuples()]
    imp["cv_auc_diff_se"] = [
        (fold_auc[(r.model, r.imputation)]
         - fold_auc[(r.model, "median")]).std(ddof=1) / np.sqrt(5)
        for r in imp.itertuples()]

    # Regla de decisión (solo CV): cambiar de mediana solo si la mejora en AUC
    # supera MIN_AUC_GAIN y 2 errores estándar de la diferencia pareada.
    chosen = {}
    for name in models:
        part = imp[imp.model == name].set_index("imputation")
        cand = part["cv_roc_auc_mean"].idxmax()
        gain = part.loc[cand, "cv_auc_diff_vs_median"]
        se = part.loc[cand, "cv_auc_diff_se"]
        chosen[name] = (cand if cand != "median" and gain > MIN_AUC_GAIN
                        and gain > 2 * se else "median")
    imp["selected"] = [chosen[r.model] == r.imputation for r in imp.itertuples()]

    # Resultado en prueba de cada estrategia (solo informativo).
    test_aucs = []
    for r in imp.itertuples():
        pipe = with_imputation(tuned[r.model], r.model, r.imputation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        test_aucs.append(classification_metrics(y_test, p)["roc_auc"])
    imp["test_roc_auc"] = test_aucs
    imp.to_csv(TABLES / "p2x_imputation_comparison.csv", index=False)
    log(f"Imputación elegida por CV: {chosen}")

    # Modelo final para C y D: el seleccionado en P2 con su imputación elegida.
    final_pipe = with_imputation(tuned[best_name], best_name, chosen[best_name])

    # ------------------------------------------------------------------
    # B. Recalibración (logística y RF usan class_weight; GB como referencia)
    # ------------------------------------------------------------------
    cal_rows, cal_probas = [], {}
    for name in ["logreg", "rf", "gboost"]:
        base = with_imputation(tuned[name], name, chosen[name])
        variants = {"none": base}
        for method in ("sigmoid", "isotonic"):
            variants[method] = CalibratedClassifierCV(
                clone(base), method=method, cv=cv)
        for method, est in variants.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # CV anidada: OOF de train para elegir el método.
                oof = cross_val_predict(clone(est), X_train, y_train, cv=cv,
                                        method="predict_proba", n_jobs=-1)[:, 1]
                fitted = clone(est).fit(X_train, y_train)
            p_test = fitted.predict_proba(X_test)[:, 1]
            cal_probas[(name, method)] = p_test
            cal_rows.append(score_row(y_train, oof, model=name,
                                      calibration=method, split="train_oof"))
            cal_rows.append(score_row(y_test, p_test, model=name,
                                      calibration=method, split="test"))
            log(f"Calibración {name:7s} {method:8s} OOF Brier "
                f"{cal_rows[-2]['brier']:.4f} | test Brier {cal_rows[-1]['brier']:.4f}")
    cal = pd.DataFrame(cal_rows)
    oof_cal = cal[cal.split == "train_oof"]
    best_cal = (oof_cal.loc[oof_cal.groupby("model")["brier"].idxmin()]
                .set_index("model")["calibration"].to_dict())
    cal["selected"] = [best_cal[r.model] == r.calibration for r in cal.itertuples()]
    cal.to_csv(TABLES / "p2x_calibration_comparison.csv", index=False)
    log(f"Calibración elegida (Brier OOF): {best_cal}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    for ax, name in zip(axes, ["logreg", "rf"]):
        ax.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1,
                label="Calibración perfecta")
        for method, color, style in (("none", MODEL_COLORS[name], "-"),
                                     ("sigmoid", "#6250d6", "-"),
                                     ("isotonic", "#e34948", "-")):
            fp, mp = calibration_curve(y_test, cal_probas[(name, method)],
                                       n_bins=10, strategy="quantile")
            label = {"none": "Original (class_weight)", "sigmoid": "Platt",
                     "isotonic": "Isotónica"}[method]
            ax.plot(mp, fp, marker="o", ms=4, lw=2, ls=style, color=color,
                    label=label)
        _style_axes(ax)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Probabilidad predicha",
               ylabel="Fracción observada de fallecidos",
               title=f"{MODEL_LABELS[name]}: recalibración (prueba)")
        ax.title.set_color(INK)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2x_recalibration.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------------
    # C. Umbral por costos (modelo final)
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = cross_val_predict(clone(final_pipe), X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        final_pipe.fit(X_train, y_train)
    p_test = final_pipe.predict_proba(X_test)[:, 1]
    joblib.dump(final_pipe, MODELS / "final_model.joblib")
    tau_f1, _ = select_threshold(y_train, oof)

    cost_rows = []
    for R in COST_RATIOS:
        tau_bayes = 1 / (1 + R)
        tau_emp, _ = select_cost_threshold(y_train, oof, R)
        for rule, tau in (("bayes_1/(1+R)", tau_bayes),
                          ("empirical_oof", tau_emp), ("max_f1_oof", tau_f1),
                          ("0.5", 0.5)):
            m = classification_metrics(y_test, p_test, tau)
            cost_rows.append({
                "cost_ratio_R": R, "rule": rule, "threshold": tau,
                "test_cost_per_patient": expected_cost(y_test, p_test, tau, R),
                "cost_treat_none": R * y_test.mean(),
                "cost_treat_all": 1 - y_test.mean(),
                **{k: m[k] for k in ("recall", "precision", "specificity",
                                     "f1", "TP", "FP", "FN", "TN")},
            })
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(TABLES / "p2x_cost_thresholds.csv", index=False)

    grid = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    ramp = ["#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
    for R, color in zip(COST_RATIOS, ramp):
        curve = [expected_cost(y_test, p_test, t, R) for t in grid]
        ax.plot(grid, curve, color=color, lw=2, label=f"R = {R}")
        ax.plot(1 / (1 + R), expected_cost(y_test, p_test, 1 / (1 + R), R),
                marker="o", ms=7, color=color, markeredgecolor=SURFACE,
                markeredgewidth=2)
    _style_axes(ax)
    ax.set(xlabel="Umbral τ", ylabel="Costo medio por paciente (C_FP = 1)",
           title="Costo esperado en prueba según R = C_FN / C_FP", xlim=(0, 1))
    ax.title.set_color(INK)
    ax.legend(fontsize=8, frameon=False, title="Puntos: τ* = 1/(1+R)",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "p2x_cost_curves.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------------
    # D. Umbral global vs. umbral por grupo diagnóstico
    # ------------------------------------------------------------------
    g_train = X_train["dzgroup"].values
    g_test = X_test["dzgroup"].values
    groups = pd.Series(g_train).value_counts().index.tolist()
    R = GROUP_COST_RATIO

    rules = {
        "global_max_f1": {g: tau_f1 for g in groups},
        f"global_cost_R{R}": {g: select_cost_threshold(y_train, oof, R)[0]
                              for g in groups},
    }
    per_f1, per_cost = {}, {}
    for g in groups:
        mask = g_train == g
        per_f1[g] = select_threshold(y_train[mask], oof[mask])[0]
        per_cost[g] = select_cost_threshold(y_train[mask], oof[mask], R)[0]
    rules["group_max_f1"] = per_f1
    rules[f"group_cost_R{R}"] = per_cost

    gt_rows, overall_rows = [], []
    for rule, taus in rules.items():
        tau_vec = np.array([taus[g] for g in g_test])
        pred = (p_test >= tau_vec).astype(int)
        # Métricas globales con umbral por paciente.
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        tn = int(((pred == 0) & (y_test == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        overall_rows.append({
            "rule": rule, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": rec, "precision": prec,
            "specificity": tn / (tn + fp),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            f"cost_per_patient_R{R}": (fp + R * fn) / len(y_test),
        })
        for g in groups:
            m = g_test == g
            yt, pr = y_test.values[m], pred[m]
            pos, neg = (yt == 1).sum(), (yt == 0).sum()
            gt_rows.append({
                "rule": rule, "dzgroup": g, "threshold": taus[g],
                "n_test": int(m.sum()), "deaths_test": int(pos),
                "fnr": ((pr == 0) & (yt == 1)).sum() / pos if pos else np.nan,
                "fpr": ((pr == 1) & (yt == 0)).sum() / neg if neg else np.nan,
            })
    pd.DataFrame(overall_rows).to_csv(
        TABLES / "p2x_group_thresholds_overall.csv", index=False)
    pd.DataFrame(gt_rows).to_csv(
        TABLES / "p2x_group_thresholds_by_group.csv", index=False)

    ext_meta = {
        "imputation_selected": chosen,
        "imputation_rule": f"AUC gain > {MIN_AUC_GAIN} and > 2 SE (paired CV)",
        "calibration_selected_by_oof_brier": best_cal,
        "final_model": best_name,
        "final_imputation": chosen[best_name],
        "final_test_roc_auc": classification_metrics(y_test, p_test)["roc_auc"],
        "max_f1_threshold": tau_f1,
        "cost_ratios": COST_RATIOS,
        "group_cost_ratio": R,
    }
    (ROOT / "results" / "p2x_run_metadata.json").write_text(
        json.dumps(ext_meta, indent=2), encoding="utf-8")
    print(json.dumps(ext_meta, indent=2))
    print(pd.DataFrame(overall_rows).round(4).to_string(index=False))
    log("Extensiones completadas.")


if __name__ == "__main__":
    main()
