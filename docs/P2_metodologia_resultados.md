# P2 — Metodología, decisiones y resultados

Documento de soporte para las secciones 4–9 del paper (Guía P2, §3). Todos los
valores provienen de `results/tables/p2_*.csv`, generados por `python scripts/run_p2.py`
(semilla 42, scikit-learn 1.9.1). Si se re-ejecuta el experimento, actualizar este
documento a partir de esas tablas.

---

## 1. Correspondencia con la guía

| Requisito P2 (Guía §2.2 / §3) | Dónde está |
|---|---|
| Preprocesamiento e ingeniería de características | `preprocessing.py`, `notebooks/02_preprocessing_baseline.ipynb`, §2 |
| Baseline | `DummyClassifier(prior)` de P1 + baseline demográfico, §3 |
| ≥ 2 modelos implementados y comparados | Reg. logística, árbol, Random Forest, Gradient Boosting, §3 y §5 |
| Diseño experimental, validación, ajuste de hiperparámetros | `scripts/run_p2.py`, §4 |
| Métricas apropiadas y justificadas | `src/evaluation.py`, §4.3 |
| Resultados con tablas y visualizaciones | `results/tables/p2_*`, `results/figures/p2_*`, §5 |
| Análisis de errores e importancia de variables | §6, §7 |
| Discusión, limitaciones, ética, conclusiones | §8, §9 |
| Reproducibilidad | `README.md`, `tests/`, `results/p2_run_metadata.json` |

## 2. Preprocesamiento (heredado de P1, extendido en P2)

| Decisión | Justificación |
|---|---|
| Partición 80/20 estratificada (7284 / 1821), semilla 42, **antes** de cualquier transformación | Evita que estadísticos del test (medianas, modas, μ, σ) contaminen el entrenamiento. Prevalencia idéntica en ambas: 0.2592. |
| Exclusión de 17 variables por fuga o posterioridad (`death`, `slos`, `d.time`, `charges`, `totcst`, `totmcst`, `avtisst`, `sfdm2`, `adlsc`, `sps`, `aps`, `surv2m`, `surv6m`, `prg2m`, `prg6m`, `dnr`, `dnrday`) | Son desenlaces, costos/recursos de toda la estadía, scores pronósticos de modelos previos u órdenes de no reanimación. |
| Exclusión de `adlp`, `urine` (> 50 % faltantes en train) | Imputar más de la mitad de valores produciría una variable mayormente artificial. |
| Numéricas: mediana; categóricas: moda + one-hot (`handle_unknown="ignore"`) | Mediana robusta a valores extremos clínicamente reales (no se eliminan outliers). |
| **Nuevo en P2:** indicadores de faltante (`add_indicator`) como hiperparámetro | En SUPPORT un laboratorio ausente suele indicar que no se solicitó; la ausencia puede ser informativa. Se decide por CV, no a priori. |
| `StandardScaler` solo en regresión logística | La penalización λ‖w‖ depende de la escala; los árboles son invariantes a transformaciones monótonas. |
| Todo el preprocesador dentro de `Pipeline` | Se reajusta en cada fold: el fold de validación nunca participa en imputación/escalado. Verificado en `tests/` y en el notebook 03 (medianas del imputador = medianas de train). |

**Grupos de variables** (`preprocessing.FEATURE_GROUPS`), usados para la pregunta de investigación:

- *Demográficas (5)*: `age`, `sex`, `race`, `income`, `edu`.
- *Clínicas (8)*: `dzgroup`, `dzclass`, `num.co`, `diabetes`, `dementia`, `ca`, `adls`, `hday`.
- *Fisiológicas, día 3 (14)*: `scoma`, `meanbp`, `wblc`, `hrt`, `resp`, `temp`, `pafi`, `alb`, `bili`, `crea`, `sod`, `ph`, `glucose`, `bun`.

## 3. Modelos

| Modelo | Formulación (diapositivas del curso) | Por qué se incluye |
|---|---|---|
| Dummy (prior) | p = P̂(y=1) constante | Piso de referencia de P1: accuracy 0.741 sin discriminar. |
| Reg. logística (solo demográficas) | igual que abajo, 5 variables | Baseline de la pregunta de investigación. |
| Regresión logística | p = σ(wᵀx+b); minimiza log-loss + λΩ(w), Ω ∈ {‖w‖₂², ‖w‖₁}, C = 1/λ | Modelo lineal interpretable (odds ratios). |
| Árbol de decisión (CART) | Gini 1−Σp̂²ₘₖ / entropía; poda R_α(T) = R(T) + α\|T\| | Reglas interpretables; ilustra sesgo-varianza. |
| Random Forest (B = 500) | bagging + submuestreo de variables; p̄(x) = (1/B) Σ P_b(y=1\|x) | Reduce la varianza del árbol. |
| Gradient Boosting | F_t(x) = F_{t−1}(x) + η h_t(x), pérdida log-loss | Reduce sesgo de forma secuencial; capta no linealidades e interacciones. |

## 4. Diseño experimental

1. **Búsqueda de hiperparámetros** — `RandomizedSearchCV` (30 candidatos para logística, 40 árbol, 20 RF, 30 GB) con `StratifiedKFold(k=5, shuffle=True, random_state=42)` **solo sobre entrenamiento**: CV(M, θ) = (1/K) Σₖ Score_k(M, θ).
2. **Métrica de selección: ROC-AUC.** No depende del umbral y no se infla con la clase mayoritaria (accuracy del Dummy = 0.741). Se registran también AP, F1, recall, precision, accuracy y log-loss.
3. **Umbral τ** (ŷ = 1[p ≥ τ]): τ = 0.5 no es un umbral clínico validado. Se elige τ que maximiza F1 sobre predicciones *out-of-fold* de entrenamiento (`cross_val_predict` con los mismos folds). El test no interviene. Se reportan τ = 0.5 y τ de CV.
4. **Modelo final**: mayor ROC-AUC medio en CV con todas las variables → **Gradient Boosting**.
5. **Test**: evaluación única al final; IC 95 % por bootstrap percentil (1000 réplicas) y diferencias **pareadas** (mismas réplicas para ambos modelos).
6. **Ablación**: la búsqueda completa se repite para regresión logística y GB con los conjuntos `demographic` y `demographic+physiological`.

### 4.1 Hiperparámetros seleccionados (CV)

| Modelo | Mejores hiperparámetros |
|---|---|
| Reg. logística | lasso (L1), C = 1.146, `class_weight="balanced"`, indicadores de faltante = sí. Lasso anula 3 de 63 coeficientes (`sex_male`, `income_>$50k`, `race_black`). |
| Árbol | Gini, `max_depth = 8`, `min_samples_leaf = 77`, `ccp_alpha ≈ 1.4·10⁻⁵`, sin indicadores. |
| Random Forest | `max_features = 0.3`, `max_depth = None`, `min_samples_leaf = 20`, `class_weight="balanced_subsample"`, indicadores = sí. |
| Gradient Boosting | η = 0.024, 233 iteraciones, 35 hojas máx., `min_samples_leaf = 37`, L2 = 1.25, `max_features = 0.5`, indicadores = sí. |

En 3 de 4 modelos la CV prefirió **añadir indicadores de faltante**, lo que respalda que la ausencia de datos es informativa.

## 5. Resultados

### 5.1 Validación cruzada (train, 5 folds)

| Modelo | AUC train | AUC CV (media ± DE) | AP CV |
|---|---|---|---|
| Dummy | 0.500 | 0.500 ± 0.000 | 0.259 |
| Reg. logística | 0.814 | 0.804 ± 0.007 | 0.609 |
| Árbol | 0.819 | 0.781 ± 0.012 | 0.558 |
| Random Forest | 0.904 | 0.818 ± 0.010 | 0.627 |
| **Gradient Boosting** | 0.947 | **0.824 ± 0.008** | **0.647** |

La brecha train−CV (GB 0.12, RF 0.09, logística 0.01) muestra la mayor varianza de los ensambles, aunque generalizan mejor que la logística. La curva de validación del árbol (`p2_tree_validation_curve.png`) muestra el compromiso sesgo-varianza: sin poda, el AUC de CV alcanza su máximo con `max_depth = 5` (0.774); a profundidad 20, el AUC de train llega a 0.999 y el de CV cae a 0.609.

### 5.2 Prueba (1821 pacientes, 472 fallecidos)

Métricas de umbral con τ elegido en CV; AUC y AP no dependen del umbral.

| Modelo | τ | Accuracy | Precision | Recall | Especificidad | F1 | ROC-AUC [IC 95 %] | AP [IC 95 %] | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Dummy (P1) | 0.50 | 0.741 | 0.000 | 0.000 | 1.000 | 0.000 | 0.500 | 0.259 | 0.192 |
| Reg. logística | 0.53 | 0.767 | 0.540 | 0.695 | 0.792 | 0.607 | 0.820 [0.800, 0.842] | 0.629 [0.583, 0.675] | 0.171 |
| Árbol | 0.31 | 0.753 | 0.518 | 0.661 | 0.785 | 0.581 | 0.799 [0.778, 0.821] | 0.575 [0.527, 0.626] | 0.149 |
| Random Forest | 0.54 | 0.778 | 0.559 | 0.674 | 0.814 | 0.611 | 0.832 [0.813, 0.853] | 0.641 [0.595, 0.688] | 0.160 |
| **Gradient Boosting** | 0.29 | 0.765 | 0.536 | 0.703 | 0.787 | 0.608 | **0.834 [0.815, 0.854]** | **0.655 [0.611, 0.700]** | **0.137** |

Con τ = 0.5, GB obtiene accuracy 0.804 pero recall 0.443: con el umbral por defecto el modelo deja sin detectar más de la mitad de los fallecimientos. Este es el motivo para fijar τ con CV.

**Diferencias pareadas de AUC frente a GB (bootstrap, test):**

| Comparación | ΔAUC | IC 95 % | Lectura |
|---|---|---|---|
| Random Forest − GB | −0.002 | [−0.008, 0.004] | No distinguibles |
| Reg. logística − GB | −0.015 | [−0.024, −0.005] | GB ligeramente superior |
| Árbol − GB | −0.035 | [−0.049, −0.020] | GB superior |

El AUC de test de GB (0.834) es coherente con su AUC de CV (0.824): no hay indicios de sobreajuste al conjunto de validación.

**Calibración** (`p2_calibration.png`): GB está bien calibrado (Brier 0.137, log-loss 0.422). La logística y RF eligieron `class_weight="balanced"`, que desplaza las probabilidades hacia la clase positiva: sobrestiman sistemáticamente el riesgo, por eso su τ óptimo está cerca de 0.5 y su Brier es peor (0.171 y 0.160) aunque discriminen bien. Si se quisieran usar sus probabilidades como riesgos absolutos, habría que recalibrarlas.

### 5.3 Pregunta de investigación: ¿fisiológicas vs. demográficas?

| Conjunto de variables | AUC test logística | AUC test GB | AUC CV logística | AUC CV GB |
|---|---|---|---|---|
| Solo demográficas | 0.567 | 0.569 | 0.570 | 0.576 |
| Demográficas + fisiológicas | 0.758 | 0.797 | 0.751 | 0.785 |
| Todas (+ clínicas) | 0.820 | 0.834 | 0.804 | 0.824 |

- Añadir las variables fisiológicas del día 3 al modelo demográfico eleva el AUC en **+0.192** (IC 95 % [0.161, 0.227]) con regresión logística y en **+0.227** con GB. El modelo completo supera al demográfico en **+0.269** [0.236, 0.304].
- Con solo variables demográficas, GB no mejora a la logística (ΔAUC +0.002, IC [−0.019, 0.024]): con 5 variables débiles no hay estructura no lineal que aprovechar.
- Con variables fisiológicas, GB supera a la logística (0.797 vs. 0.758): las relaciones fisiológicas con la mortalidad no son lineales (por ejemplo, riesgo en ambos extremos de presión arterial o temperatura).
- Diagnóstico, comorbilidad y estado funcional aportan un incremento adicional menor (+0.04 a +0.06).

**Respuesta:** las variables demográficas por sí solas apenas discriminan (AUC ≈ 0.57, cerca del azar). Las fisiológicas aportan la mayor parte de la capacidad predictiva, lo que confirma la hipótesis de P1.

## 6. Interpretación

**Importancia por permutación** (GB, caída de AUC en test, 20 repeticiones; `p2_permutation_importance.png`):

| Variable | Grupo | ΔAUC |
|---|---|---|
| `scoma` (coma de Glasgow, SUPPORT) | fisiológica | 0.073 |
| `dzclass` | clínica | 0.027 |
| `hday` | clínica | 0.024 |
| `age` | demográfica | 0.014 |
| `dzgroup` | clínica | 0.007 |
| `glucose`, `meanbp`, `bun`, `bili`, `crea`, `hrt` | fisiológicas | 0.004–0.007 c/u |

Suma por grupo: fisiológicas 0.113, clínicas 0.062, demográficas 0.017. `sex`, `income`, `race`, `diabetes` y `dementia` aportan ≈ 0 (en la logística, lasso anula `sex_male`, `race_black` e `income_>$50k`). La permutación no se reparte bien entre variables correlacionadas (`dzgroup`/`dzclass`), así que la importancia de cada una individual se subestima.

**Coeficientes de la regresión logística** (`p2_logreg_coefficients.csv`, por 1 DE o frente a la categoría omitida): los mayores son el grupo diagnóstico (MOSF con malignidad OR = 2.02, cirrosis OR = 1.90; ICC OR = 0.51, cáncer de colon OR = 0.54) y `scoma` (OR = 1.80 por DE). Son asociaciones predictivas condicionadas al resto de variables, **no** efectos causales.

## 7. Análisis de errores (GB, τ = 0.293, test)

Resultado: TP = 332, FN = 140, FP = 288, TN = 1061.

**Perfil de los errores** (`p2_error_profile.csv`, medianas): los falsos negativos son fallecidos que el día 3 parecían estables: `scoma` = 0 (vs. 37 en TP), `hday` = 1 (vs. 5), frecuencia cardíaca 90 (vs. 110), temperatura 36.7 (vs. 37.5). El modelo no puede anticipar deterioros posteriores a las mediciones disponibles. Los falsos positivos comparten el perfil de gravedad de los TP (FC 109, `hday` 4).

**Por diagnóstico** (`p2_subgroup_performance.csv`): buena parte de la discriminación global ocurre *entre* grupos diagnósticos. Dentro de cada grupo el AUC es menor (0.69–0.81, salvo cáncer de colon, con n = 99 y 6 muertes), y el umbral único produce errores asimétricos:

| Grupo | n | Mortalidad | FNR | FPR |
|---|---|---|---|---|
| ICC (CHF) | 272 | 5.9 % | 0.81 | 0.004 |
| EPOC | 201 | 13.4 % | 0.78 | 0.04 |
| Cáncer de pulmón | 170 | 7.1 % | 0.83 | 0.006 |
| MOSF con malignidad | 137 | 49.6 % | 0.03 | 0.84 |
| Coma | 119 | 57.1 % | 0.07 | 0.78 |

En los grupos de baja mortalidad casi no se detectan muertes; en los de alta mortalidad casi todos los pacientes reciben una alerta.

**Por sexo, raza, edad e ingreso**: AUC mujeres 0.849 vs. hombres 0.825 (FNR 0.26 vs. 0.32); blancos 0.835 vs. negros 0.802 (FNR 0.30 vs. 0.34). Por edad, el AUC es estable (0.83–0.84). Los grupos `asian` (n = 17), `other` (n = 14) y raza faltante (n = 9) son demasiado pequeños para extraer conclusiones. Estas diferencias son descriptivas y no se probó su significancia.

**Faltantes**: los TP tienen más glucosa/BUN faltantes que los FN (57 % vs. 41 %), lo que es coherente con que la CV haya elegido indicadores de faltante.

## 8. Discusión y limitaciones

- **Por qué ganan los ensambles de árboles.** Capturan no linealidades (riesgo en ambos extremos de signos vitales) e interacciones diagnóstico × fisiología sin ingeniería manual. Boosting reduce sesgo y bagging varianza; el árbol individual queda último por su alta varianza. La ventaja de GB sobre la logística es pequeña (0.015 AUC), y la logística sigue siendo competitiva y más interpretable.
- **Temporalidad.** Las mediciones son del día 3 *del estudio*, no de la hospitalización (`hday` > 1 en 3157 pacientes). 138 fallecidos tienen `slos = 3` y `d.time = 3`, es decir, mediciones muy cercanas al desenlace (`temporal_audit.csv`). Además `hday` es la tercera variable más importante: refleja cuánto tiempo llevaba el paciente hospitalizado al ser incluido, un artefacto del diseño del estudio que puede no estar disponible del mismo modo en otro contexto. Es una clasificación **retrospectiva**, no una validación prospectiva.
- **Umbral.** El τ que maximiza F1 trata igual FN y FP. El análisis con costos explícitos (§10C) muestra cómo cambia el umbral según R, pero los valores de R son supuestos de sensibilidad, no costos clínicos medidos. Los umbrales por grupo no mejoran el desempeño global (§10D).
- **Optimismo residual.** τ se elige con los mismos folds que los hiperparámetros (dentro de train), lo que puede ser levemente optimista. El test, independiente, lo acota.
- **Validez externa.** SUPPORT (1989–1994, 5 hospitales de EE. UU., pacientes gravemente enfermos) no representa la práctica actual ni otras poblaciones; no hay validación externa.
- **Datos faltantes.** La imputación por mediana es simple, pero las alternativas evaluadas (valores normales de SUPPORT, KNN e iterativa; §10A) no la mejoran de forma relevante. No se evaluó la imputación múltiple con combinación de estimaciones (reglas de Rubin).
- **Ética.** Un modelo de mortalidad puede influir en decisiones de limitación del esfuerzo terapéutico. Por eso no debe usarse como decisor autónomo, y las diferencias de error por raza y sexo deben auditarse con muestras mayores antes de cualquier uso. Las variables `dnr`/`dnrday` se excluyeron precisamente para no aprender decisiones médicas sobre el final de la vida.

## 9. Conclusiones

1. Todos los modelos superan ampliamente al baseline (AUC 0.80–0.83 vs. 0.50). El modelo final, Gradient Boosting, obtiene AUC 0.834 [0.815, 0.854], AP 0.655 y, con τ = 0.29, recall 0.70 y precisión 0.54, con buena calibración.
2. Las variables fisiológicas del día 3 explican la mayor parte del poder predictivo: +0.19 a +0.23 AUC sobre un modelo solo demográfico (≈ 0.57). `scoma` es el predictor individual más fuerte.
3. RF y GB son estadísticamente indistinguibles en el test; la regresión logística queda ligeramente por debajo, pero es interpretable.
4. La imputación tiene un efecto mínimo; la recalibración es necesaria cuando se usan pesos de clase; y con probabilidades calibradas el umbral por costo sigue τ* = 1/(1+R), mientras que los umbrales por grupo diagnóstico no mejoran el desempeño global (§10).
5. **Trabajo futuro:** definir los costos de error con personal clínico; mejorar la discriminación dentro de los grupos de baja mortalidad; incorporar información temporal; validación externa o por hospital (si se dispusiera del identificador).

## 10. Extensiones: imputación, recalibración y umbrales

Generadas por `python scripts/run_p2_extensions.py` (tablas `results/tables/p2x_*`, figuras `results/figures/p2x_*`). Parten de los modelos ya ajustados; todas las decisiones se toman en entrenamiento.

**A. Imputación** (CV de 5 folds, hiperparámetros fijos). Estrategias: mediana; valores normales de la documentación oficial de SUPPORT (alb 3.5, pafi 333.3, bili 1.01, crea 1.01, bun 6.51, wblc 9; mediana para el resto); KNN (k = 10); iterativa tipo MICE. KNN e iterativa se aplican tras estandarizar. Regla fijada de antemano: cambiar de mediana solo si la mejora en AUC de CV supera 0.005 **y** 2 errores estándar de la diferencia pareada.

| Modelo | Mediana | Normales | KNN | Iterativa |
|---|---|---|---|---|
| Reg. logística | 0.8044 | 0.8043 | 0.8048 | 0.8041 |
| Árbol | 0.7813 | 0.7789 | 0.7815 | 0.7748 |
| Random Forest | 0.8184 | 0.8198 | 0.8184 | 0.8178 |
| Gradient Boosting | 0.8235 | 0.8258 | 0.8245 | 0.8220 |

La mayor mejora (valores normales en GB, +0.0023, EE 0.0009) no supera 0.005, así que **se conserva la mediana**. El desempeño es robusto a la imputación.

**B. Recalibración** (`CalibratedClassifierCV`, 5 folds dentro de train; método elegido por el menor Brier OOF en CV anidada):

| Modelo | Método elegido | Brier test (antes → después) | ECE test (antes → después) | AUC test |
|---|---|---|---|---|
| Reg. logística | Platt | 0.171 → 0.141 | 0.166 → 0.024 | 0.820 → 0.820 |
| Random Forest | Isotónica | 0.160 → 0.139 | 0.140 → 0.026 | 0.832 → 0.830 |
| Gradient Boosting | Platt | 0.136 → 0.135 | 0.020 → 0.024 | 0.834 → 0.838 |

La recalibración corrige la sobrestimación de riesgo que introduce `class_weight="balanced"` sin afectar la discriminación. GB ya estaba calibrado y el cambio es marginal, por lo que el análisis de umbrales usa el modelo final sin recalibrar.

**C. Umbral con costos explícitos.** Con C_FP = 1 y C_FN = R, la regla de Bayes da τ* = 1/(1+R). No existe un R validado para esta población, así que se hace un análisis de sensibilidad. τ̂ minimiza el costo en predicciones OOF de train; métricas en prueba:

| R | τ* | τ̂ | Recall | Precision | Especificidad | Costo/paciente (τ̂) | Mejor trivial |
|---|---|---|---|---|---|---|---|
| 1 | 0.500 | 0.55 | 0.381 | 0.729 | 0.950 | 0.197 | 0.259 |
| 2 | 0.333 | 0.30 | 0.701 | 0.547 | 0.797 | 0.305 | 0.518 |
| 3 | 0.250 | 0.25 | 0.752 | 0.502 | 0.739 | 0.386 | 0.741 |
| 5 | 0.167 | 0.16 | 0.886 | 0.431 | 0.591 | 0.451 | 0.741 |
| 10 | 0.091 | 0.08 | 0.975 | 0.352 | 0.373 | 0.530 | 0.741 |

τ* ≈ τ̂ porque GB está calibrado. El modelo tiene menor costo que no alertar a nadie o alertar a todos en todos los valores de R.

**D. Umbral por grupo diagnóstico** (`dzgroup`, elegido en OOF de train) frente al umbral global, en prueba:

| Regla | Recall | Precision | F1 | Costo/paciente (R = 3) |
|---|---|---|---|---|
| Global, máx. F1 | 0.703 | 0.535 | 0.608 | 0.389 |
| Por grupo, máx. F1 | 0.737 | 0.500 | 0.596 | 0.395 |
| Global, costo R = 3 | 0.752 | 0.502 | 0.602 | 0.386 |
| Por grupo, costo R = 3 | 0.750 | 0.504 | 0.603 | 0.386 |

Los umbrales por grupo redistribuyen errores (con F1, el FNR en EPOC baja de 0.78 a 0.37 a cambio de un FPR de 0.22), pero no mejoran el desempeño global. Es lo esperable con probabilidades calibradas, cuyo umbral de costo óptimo es el mismo para todos los grupos. Además, son inestables en grupos pequeños (EPOC: τ = 0.13 con F1 vs. 0.37 con costo). La limitación real es la discriminación dentro de cada grupo.

## 11. Reproducción

```bash
pip install -r requirements.txt
python -m pytest -q             # 21 pruebas de integridad, imputación y no-fuga
python scripts/run_p2.py                # ~5 min; regenera results/tables|figures/p2_*
python scripts/run_p2_extensions.py     # ~8 min; imputación, recalibración, umbrales (p2x_*)
python scripts/make_paper_assets.py     # tablas LaTeX y figuras en paper/
python scripts/run_p2.py --quick        # prueba rápida → results/quick/ (no versionado)
```

Después, abrir `notebooks/03_modeling_evaluation.ipynb`.
