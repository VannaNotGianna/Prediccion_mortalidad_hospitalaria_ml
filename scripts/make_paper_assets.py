"""Genera las tablas LaTeX y figuras del paper a partir de results/.

Uso (desde la raíz, después de run_p2.py y run_p2_extensions.py):

    python scripts/make_paper_assets.py

Escribe ``paper/tables/*.tex`` (incluidas con \\input en el paper) y copia /
genera las figuras en ``paper/figures/p2_*.png``. Ningún número del paper
se escribe a mano en las tablas. Las figuras se copian a ``paper/figures``
para que la carpeta ``paper/`` sea autocontenida (p. ej., para Overleaf).
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # raíz del repositorio
sys.path.insert(0, str(ROOT))

from src.evaluation import MODEL_COLORS, plot_barh  # noqa: E402
T = ROOT / "results" / "tables"
F = ROOT / "results" / "figures"
PT = ROOT / "paper" / "tables"
PF = ROOT / "paper" / "figures"

NAMES = {"dummy": "Dummy (prior)", "logreg": "Reg. logística",
         "tree": "Árbol de decisión", "rf": "Random Forest",
         "gboost": "Gradient Boosting"}
SETS = {"demographic": "Demográficas",
        "demographic+physiological": "Demog. + fisiológicas",
        "full": "Todas"}
IMP = {"median": "Mediana", "normal": "Normales", "knn": "KNN",
       "iterative": "Iterativa"}
CAL = {"none": "Sin recalibrar", "sigmoid": "Platt", "isotonic": "Isotónica"}


def f3(x):
    return f"{x:.3f}"


def table(rows, header, colspec, caption, label, wide=False, size="footnotesize"):
    env = "table*" if wide else "table"
    # Las tablas anchas pueden ir a una página de floats ([p]) para no dejar
    # líneas sueltas de texto bajo varios floats de dos columnas.
    placement = "!tp" if wide else "!t"
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    tex = (f"\\begin{{{env}}}[{placement}]\n\\centering\n\\{size}\n"
           f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\setlength{{\\tabcolsep}}{{4pt}}\n"
           f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
           f"{' & '.join(header)} \\\\\n\\midrule\n{body} \\\\\n"
           f"\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")
    return tex


def write(name, tex):
    (PT / name).write_text(tex, encoding="utf-8")
    print("tabla:", name)


def main():
    PT.mkdir(parents=True, exist_ok=True)
    PF.mkdir(parents=True, exist_ok=True)

    meta = json.loads((ROOT / "results" / "p2_run_metadata.json").read_text("utf-8"))
    best = meta["selected_model"]

    # --- Validación cruzada --------------------------------------------------
    cv = pd.read_csv(T / "p2_cv_results.csv")
    full = cv[cv.feature_set == "full"].set_index("model")
    rows = []
    for m in ["dummy", "logreg", "tree", "rf", "gboost"]:
        r = full.loc[m]
        name = f"\\textbf{{{NAMES[m]}}}" if m == best else NAMES[m]
        rows.append([name, f3(r.train_roc_auc),
                     f"{r.cv_roc_auc_mean:.3f} $\\pm$ {r.cv_roc_auc_std:.3f}",
                     f3(r.cv_average_precision_mean),
                     f"{-r.cv_neg_log_loss_mean:.3f}"])
    write("tab_cv.tex", table(
        rows, ["Modelo", "AUC train", "AUC CV", "AP CV", "Log-loss CV"],
        "lcccc",
        "Validación cruzada estratificada (5 folds, solo entrenamiento) del "
        "mejor candidato de cada modelo. En negrita, el modelo seleccionado.",
        "tab:cv"))

    # --- Hiperparámetros -----------------------------------------------------
    hp = {
        "logreg": ("$C\\in[10^{-3},10^{2}]$, L1/L2, \\texttt{class\\_weight}",
                   lambda p: f"L{'1' if p['classifier__l1_ratio'] == 1.0 else '2'}, "
                             f"$C$={float(p['classifier__C']):.2f}, "
                             f"{p['classifier__class_weight']}"),
        "tree": ("criterio, profundidad, hoja mín., $\\alpha$ de poda",
                 lambda p: f"{p['classifier__criterion']}, prof.={p['classifier__max_depth']}, "
                           f"hoja={p['classifier__min_samples_leaf']}, "
                           f"$\\alpha$={float(p['classifier__ccp_alpha']):.1e}"),
        "rf": ("\\texttt{max\\_features}, profundidad, hoja mín., pesos",
               lambda p: f"feat.={p['classifier__max_features']}, "
                         f"prof.={p['classifier__max_depth']}, "
                         f"hoja={p['classifier__min_samples_leaf']}, "
                         f"{p['classifier__class_weight']}"),
        "gboost": ("$\\eta$, iteraciones, hojas, hoja mín., L2",
                   lambda p: f"$\\eta$={float(p['classifier__learning_rate']):.3f}, "
                             f"iter.={p['classifier__max_iter']}, "
                             f"hojas={p['classifier__max_leaf_nodes']}, "
                             f"L2={float(p['classifier__l2_regularization']):.2f}"),
    }
    rows = []
    for m, (space, fmt) in hp.items():
        p = json.loads(full.loc[m, "best_params"])
        ind = "sí" if p["preprocessor__numeric__imputer__add_indicator"] else "no"
        rows.append([NAMES[m], space,
                     fmt(p).replace("_", "\\_").replace("None", "ninguno"), ind])
    write("tab_hparams.tex", table(
        rows, ["Modelo", "Espacio de búsqueda", "Seleccionado (CV)", "Ind. faltantes"],
        "lp{5.2cm}p{6.6cm}c",
        "Hiperparámetros explorados con \\texttt{RandomizedSearchCV} y valores "
        "seleccionados por ROC-AUC en validación cruzada.",
        "tab:hparams", wide=True))

    # --- Prueba ----------------------------------------------------------------
    test = pd.read_csv(T / "p2_test_metrics.csv")
    boot = pd.read_csv(T / "p2_bootstrap_vs_best.csv").set_index("model")
    rows = []
    for m in ["dummy", "logreg", "tree", "rf", "gboost"]:
        rule = "0.5" if m == "dummy" else "cv_f1"
        r = test[(test.model == m) & (test.feature_set == "full")
                 & (test.threshold_rule == rule)].iloc[0]
        if m == "dummy":
            auc, ap = "0.500", f3(r.average_precision)
        else:
            b = boot.loc[f"{m}|full"]
            auc = f"{b.roc_auc:.3f} [{b.roc_auc_ci_low:.3f}, {b.roc_auc_ci_high:.3f}]"
            ap = f"{b.average_precision:.3f} [{b.average_precision_ci_low:.3f}, " \
                 f"{b.average_precision_ci_high:.3f}]"
        name = f"\\textbf{{{NAMES[m]}}}" if m == best else NAMES[m]
        rows.append([name, f"{r.threshold:.2f}", f3(r.accuracy), f3(r.precision),
                     f3(r.recall), f3(r.specificity), f3(r.f1), auc, ap,
                     f3(r.brier), f"{r.TP}/{r.FP}/{r.FN}/{r.TN}"])
    write("tab_test.tex", table(
        rows, ["Modelo", "$\\tau$", "Acc.", "Prec.", "Recall", "Espec.", "F1",
               "ROC-AUC [IC 95\\%]", "AP [IC 95\\%]", "Brier", "TP/FP/FN/TN"],
        "lcccccccccc",
        "Resultados en el conjunto de prueba ($n=1821$, 472 fallecidos). "
        "$\\tau$ se eligió maximizando F1 en predicciones \\textit{out-of-fold} "
        "de entrenamiento (0.5 para el Dummy). IC por bootstrap (1000 réplicas).",
        "tab:test", wide=True, size="scriptsize"))

    # --- Ablación de variables -------------------------------------------------
    bd = pd.read_csv(T / "p2_bootstrap_vs_demographic.csv").set_index("model")
    cvi = cv.set_index(["model", "feature_set"])
    short = {"demographic": "Demog.", "demographic+physiological": "Demog.+fisiol.",
             "full": "Todas"}
    rows = []
    for fs in ["demographic", "demographic+physiological", "full"]:
        row = [short[fs]]
        for m in ["logreg", "gboost"]:
            row += [f3(cvi.loc[(m, fs), "cv_roc_auc_mean"]),
                    f3(bd.loc[f"{m}|{fs}", "roc_auc"])]
        key = f"logreg|{fs}"
        if fs == "demographic":
            row.append("---")
        else:
            r = bd.loc[key]
            row.append(f"+{r['roc_auc_diff_vs_logreg|demographic']:.3f} "
                       f"[{r.roc_auc_diff_ci_low:.2f}, {r.roc_auc_diff_ci_high:.2f}]")
        rows.append(row)
    write("tab_ablation.tex", table(
        rows, ["Variables", "RL CV", "RL test", "GB CV", "GB test",
               "$\\Delta$AUC RL"],
        "lccccc",
        "ROC-AUC por conjunto de variables (RL: regresión logística; GB: "
        "Gradient Boosting). $\\Delta$AUC RL: diferencia pareada en prueba frente "
        "a la RL solo demográfica [IC 95\\,\\%].",
        "tab:ablation", size="scriptsize"))

    # --- Imputación ------------------------------------------------------------
    imp = pd.read_csv(T / "p2x_imputation_comparison.csv")
    rows = []
    for m in ["logreg", "tree", "rf", "gboost"]:
        part = imp[imp.model == m].set_index("imputation")
        best_cv = part.cv_roc_auc_mean.idxmax()
        cells = []
        for s in ["median", "normal", "knn", "iterative"]:
            v = f3(part.loc[s, "cv_roc_auc_mean"])
            cells.append(f"\\textbf{{{v}}}" if s == best_cv else v)
        rows.append([NAMES[m]] + cells)
    write("tab_imputation.tex", table(
        rows, ["Modelo"] + [IMP[s] for s in ["median", "normal", "knn", "iterative"]],
        "lcccc",
        "ROC-AUC medio en CV (5 folds, entrenamiento) según la imputación "
        "numérica, con los hiperparámetros ya seleccionados. En negrita, el máximo "
        "por modelo.",
        "tab:imputation"))

    # --- Recalibración ---------------------------------------------------------
    cal = pd.read_csv(T / "p2x_calibration_comparison.csv")
    rows = []
    for m in ["logreg", "rf", "gboost"]:
        for c in ["none", "sigmoid", "isotonic"]:
            o = cal[(cal.model == m) & (cal.calibration == c) & (cal.split == "train_oof")].iloc[0]
            t = cal[(cal.model == m) & (cal.calibration == c) & (cal.split == "test")].iloc[0]
            mark = "$^{*}$" if o.selected else ""
            rows.append([NAMES[m] if c == "none" else "", CAL[c] + mark,
                         f3(o.brier), f3(t.brier), f3(t.ece), f3(t.log_loss),
                         f3(t.roc_auc)])
    write("tab_calibration.tex", table(
        rows, ["Modelo", "Calibración", "Brier OOF", "Brier test", "ECE test",
               "Log-loss test", "AUC test"],
        "llccccc",
        "Recalibración con \\texttt{CalibratedClassifierCV} (5 folds dentro de "
        "entrenamiento). $^{*}$Método elegido por menor Brier \\textit{out-of-fold} "
        "en entrenamiento.",
        "tab:calibration", wide=True))

    # --- Umbrales por costo ----------------------------------------------------
    costs = pd.read_csv(T / "p2x_cost_thresholds.csv")
    rows = []
    for R in sorted(costs.cost_ratio_R.unique()):
        b = costs[(costs.cost_ratio_R == R) & (costs.rule == "bayes_1/(1+R)")].iloc[0]
        e = costs[(costs.cost_ratio_R == R) & (costs.rule == "empirical_oof")].iloc[0]
        rows.append([str(R), f"{b.threshold:.3f}", f"{e.threshold:.2f}",
                     f3(e.recall), f3(e.precision), f3(e.specificity),
                     f3(b.test_cost_per_patient), f3(e.test_cost_per_patient),
                     f3(min(b.cost_treat_none, b.cost_treat_all))])
    write("tab_costs.tex", table(
        rows, ["$R$", "$\\tau^{*}$", "$\\hat\\tau$", "Recall", "Prec.", "Espec.",
               "Costo $\\tau^{*}$", "Costo $\\hat\\tau$", "Costo trivial"],
        "ccccccccc",
        "Umbral con costos explícitos ($C_{FP}=1$, $C_{FN}=R$) para el modelo "
        "final. $\\tau^{*}=1/(1+R)$ (regla de Bayes); $\\hat\\tau$ minimiza el "
        "costo en predicciones OOF de entrenamiento. Métricas y costo medio por "
        "paciente en prueba (con $\\hat\\tau$). Costo trivial: mejor entre no "
        "alertar a nadie o alertar a todos.",
        "tab:costs", wide=True))

    # --- Umbrales por grupo ----------------------------------------------------
    xm = json.loads((ROOT / "results" / "p2x_run_metadata.json").read_text("utf-8"))
    R = xm["group_cost_ratio"]
    gb = pd.read_csv(T / "p2x_group_thresholds_by_group.csv")
    go = pd.read_csv(T / "p2x_group_thresholds_overall.csv").set_index("rule")
    rules = ["global_max_f1", "group_max_f1", f"global_cost_R{R}", f"group_cost_R{R}"]
    rows = []
    for g in gb[gb.rule == rules[0]].sort_values("n_test", ascending=False).dzgroup:
        row = [g.replace("&", "\\&"),
               str(int(gb[(gb.rule == rules[0]) & (gb.dzgroup == g)].n_test.iloc[0]))]
        for rule in rules:
            r = gb[(gb.rule == rule) & (gb.dzgroup == g)].iloc[0]
            row.append(f"{r.threshold:.2f} / {r.fnr:.2f} / {r.fpr:.2f}")
        rows.append(row)
    total = ["\\textit{Global}", "1821"]
    for rule in rules:
        o = go.loc[rule]
        total.append(f"Rec {o.recall:.2f} / F1 {o.f1:.2f} / C {o[f'cost_per_patient_R{R}']:.3f}")
    rows.append(total)
    write("tab_groups.tex", table(
        rows, ["Grupo", "$n$", "Global F1", "Por grupo F1",
               f"Global costo $R$={R}", f"Por grupo costo $R$={R}"],
        "lrcccc",
        "Umbral global frente a umbral por grupo diagnóstico (modelo final, prueba). "
        "Cada celda: $\\tau$ / FNR / FPR. Última fila: recall, F1 y costo medio "
        f"por paciente ($R$={R}). Todos los umbrales se eligieron en predicciones "
        "OOF de entrenamiento.",
        "tab:groups", wide=True, size="scriptsize"))

    # --- Figuras -------------------------------------------------------------
    for name in ["p2_roc_pr_curves", "p2_confusion_matrices",
                 "p2_tree_validation_curve", "p2x_recalibration",
                 "p2x_cost_curves"]:
        shutil.copy(F / f"{name}.png", PF / f"{name}.png")
        print("figura:", name)
    perm = pd.read_csv(T / "p2_permutation_importance.csv").head(15)
    plot_barh(perm.set_index("feature")["importance_mean"],
              perm.set_index("feature")["importance_std"],
              f"Importancia por permutación (15 principales) — {NAMES[best]}",
              "Disminución media de ROC-AUC en prueba",
              PF / "p2_permutation_importance_top15.png",
              color=MODEL_COLORS[best], reference_line=0)
    print("figura: p2_permutation_importance_top15")


if __name__ == "__main__":
    main()
