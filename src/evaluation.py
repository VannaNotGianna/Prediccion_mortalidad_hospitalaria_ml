"""Métricas, selección de umbral, intervalos bootstrap y figuras (P2).

Convención: la clase positiva es ``hospdead = 1`` (fallecimiento
hospitalario). Matriz de confusión con filas = real, columnas = predicho.

Fórmulas (Formulario P2):
    Accuracy    = (TP + TN) / N
    Precision   = TP / (TP + FP)
    Recall/TPR  = TP / (TP + FN)
    Specificity = TN / (TN + FP) = 1 − FPR
    F1          = 2·P·R / (P + R)
    Log-loss    = −(1/n) Σ [y ln p + (1−y) ln(1−p)]
    ŷ           = 1[p ≥ τ]
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Paleta categórica validada (orden fijo; el color sigue a la entidad).
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "logreg": "#2a78d6",
    "tree": "#eb6834",
    "rf": "#1baf7a",
    "gboost": "#eda100",
    "logreg_demographic": "#e87ba4",
    "dummy": "#8a8985",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def classification_metrics(y_true, proba, threshold=0.5):
    """Métricas de umbral y de ranking para la clase positiva."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def _safe(num, den):
        return num / den if den else 0.0

    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    has_both = len(np.unique(y_true)) == 2

    return {
        "threshold": float(threshold),
        "accuracy": _safe(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": _safe(tn, tn + fp),
        "f1": _safe(2 * precision * recall, precision + recall),
        "roc_auc": roc_auc_score(y_true, proba) if has_both else np.nan,
        "average_precision": (
            average_precision_score(y_true, proba) if has_both else np.nan
        ),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15),
                             labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def expected_calibration_error(y_true, proba, n_bins=10):
    """ECE = Σ_b (n_b/N)·|media(p)_b − fracción observada_b|, bins de igual ancho."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(proba[mask].mean() - y_true[mask].mean())
    return float(ece)


def expected_cost(y_true, proba, threshold, cost_ratio):
    """Costo medio por paciente con C_FP = 1 y C_FN = cost_ratio."""
    m = classification_metrics(y_true, proba, threshold)
    return (m["FP"] + cost_ratio * m["FN"]) / len(np.asarray(y_true))


def select_cost_threshold(y_true, proba, cost_ratio, grid=None):
    """Umbral que minimiza FP + R·FN (empírico, sobre predicciones OOF de train)."""
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    costs = [expected_cost(y_true, proba, t, cost_ratio) for t in grid]
    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best])


def select_threshold(y_true, proba):
    """Umbral τ que maximiza F1 sobre predicciones *out-of-fold* de train.

    Nunca debe llamarse con datos de prueba (regla RB-09).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def bootstrap_ci(y_true, probas, reference=None, n_boot=1000, seed=42):
    """IC 95% bootstrap (percentil) de ROC-AUC y AP en el conjunto de prueba.

    ``probas`` es ``{nombre: proba}``. Si se indica ``reference``, se calcula
    también la diferencia pareada (modelo − referencia) con las mismas
    réplicas, lo que permite comparar modelos sobre los mismos pacientes.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)

    samples = {name: {"roc_auc": [], "average_precision": []} for name in probas}
    diffs = {name: {"roc_auc": [], "average_precision": []} for name in probas}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.min() == yb.max():
            continue
        ref_vals = None
        if reference is not None:
            pr = np.asarray(probas[reference])[idx]
            ref_vals = {
                "roc_auc": roc_auc_score(yb, pr),
                "average_precision": average_precision_score(yb, pr),
            }
        for name, p in probas.items():
            pb = np.asarray(p)[idx]
            vals = {
                "roc_auc": roc_auc_score(yb, pb),
                "average_precision": average_precision_score(yb, pb),
            }
            for metric, value in vals.items():
                samples[name][metric].append(value)
                if ref_vals is not None:
                    diffs[name][metric].append(value - ref_vals[metric])

    rows = []
    for name, p in probas.items():
        row = {"model": name}
        for metric, scorer in (("roc_auc", roc_auc_score),
                               ("average_precision", average_precision_score)):
            low, high = np.percentile(samples[name][metric], [2.5, 97.5])
            row[metric] = scorer(y_true, p)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            if reference is not None and name != reference:
                d = np.asarray(diffs[name][metric])
                dl, dh = np.percentile(d, [2.5, 97.5])
                row[f"{metric}_diff_vs_{reference}"] = d.mean()
                row[f"{metric}_diff_ci_low"] = dl
                row[f"{metric}_diff_ci_high"] = dh
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_table(X, y_true, proba, threshold, column, bins=None, labels=None):
    """Desempeño por subgrupo (análisis de errores y sesgo)."""
    values = X[column]
    if bins is not None:
        values = pd.cut(values, bins=bins, labels=labels, right=False)
    values = values.astype("object").fillna("faltante")

    frame = pd.DataFrame({
        "group": values.values,
        "y": np.asarray(y_true),
        "p": np.asarray(proba),
    })

    rows = []
    for group, part in frame.groupby("group", sort=False):
        m = classification_metrics(part["y"], part["p"], threshold)
        rows.append({
            "variable": column,
            "group": group,
            "n": len(part),
            "prevalence": part["y"].mean(),
            "mean_predicted": part["p"].mean(),
            "roc_auc": m["roc_auc"],
            "recall": m["recall"],
            "precision": m["precision"],
            "fnr": 1 - m["recall"] if (m["TP"] + m["FN"]) else np.nan,
            "fpr": 1 - m["specificity"] if (m["TN"] + m["FP"]) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def plot_roc_pr(y_true, probas, labels, prevalence, path):
    """Curvas ROC y Precision-Recall de todos los modelos (test)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)

    for name, p in probas.items():
        color = MODEL_COLORS.get(name, INK_2)
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{labels[name]} (AUC={auc:.3f})")
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f"{labels[name]} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1,
                 label="Azar (AUC=0.5)")
    axes[1].axhline(prevalence, ls="--", color=INK_2, lw=1,
                    label=f"Prevalencia ({prevalence:.3f})")

    axes[0].set(xlabel="Tasa de falsos positivos (1 − especificidad)",
                ylabel="Recall (TPR)", title="Curva ROC — prueba")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Curva Precision-Recall — prueba")
    for ax in axes:
        _style_axes(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.title.set_color(INK)
        ax.legend(fontsize=8, frameon=False, loc="lower right"
                  if ax is axes[0] else "upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrices(matrices, path):
    """Matrices de confusión (filas = real, columnas = predicho)."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.3), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])

    for ax, (title, cm) in zip(axes, matrices.items()):
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.sum())
        for (i, j), v in np.ndenumerate(cm):
            share = v / cm.sum()
            ax.text(j, i, f"{v}\n({share:.1%})", ha="center", va="center",
                    fontsize=9, color="white" if share > 0.35 else INK)
        ax.set_xticks([0, 1], ["Pred. 0", "Pred. 1"])
        ax.set_yticks([0, 1], ["Real 0", "Real 1"])
        ax.set_title(title, fontsize=9, color=INK)
        ax.tick_params(labelsize=8, colors=INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_calibration(y_true, probas, labels, path, n_bins=10):
    """Curva de calibración (fiabilidad) por cuantiles."""
    fig, ax = plt.subplots(figsize=(5.4, 4.8), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1,
            label="Calibración perfecta")
    for name, p in probas.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=4, lw=2,
                color=MODEL_COLORS.get(name, INK_2), label=labels[name])
    _style_axes(ax)
    ax.set(xlabel="Probabilidad predicha", ylabel="Fracción observada de fallecidos",
           title="Calibración — prueba", xlim=(0, 1), ylim=(0, 1))
    ax.title.set_color(INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_barh(values, errors, title, xlabel, path, color="#2a78d6",
              reference_line=None):
    """Barras horizontales ordenadas (importancias, coeficientes)."""
    values = pd.Series(values)
    order = values.sort_values().index
    fig, ax = plt.subplots(figsize=(6.4, 0.28 * len(values) + 1.2),
                           facecolor=SURFACE)
    ax.barh(range(len(order)), values[order],
            xerr=None if errors is None else pd.Series(errors)[order],
            color=color, height=0.7, error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_yticks(range(len(order)), order)
    if reference_line is not None:
        ax.axvline(reference_line, color=INK_2, lw=1)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=xlabel, title=title)
    ax.title.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_validation_curve(param_values, train_scores, val_scores, xlabel,
                          title, path):
    """Curva de validación (sesgo-varianza) con media ± 1 desviación."""
    x = np.arange(len(param_values))
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor=SURFACE)
    for scores, label, color in ((train_scores, "Entrenamiento", "#2a78d6"),
                                 (val_scores, "Validación (CV)", "#eb6834")):
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(x, mean, marker="o", ms=5, lw=2, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15,
                        linewidth=0)
    ax.set_xticks(x, [str(v) for v in param_values])
    _style_axes(ax)
    ax.set(xlabel=xlabel, ylabel="ROC-AUC", title=title)
    ax.title.set_color(INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
