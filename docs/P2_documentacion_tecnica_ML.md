# Especificación técnica y análisis de diapositivas — Proyecto P2 de Machine Learning

**Fuente:** nueve presentaciones de Jan Hurtado, UTEC. **Alcance:** teoría, fórmulas, supuestos, transformaciones y elementos visuales del material proporcionado. **Estado:** documento de referencia; no equivale a las instrucciones oficiales de entrega de P2 ni a un esquema real de `support2.csv`.

## 1. Convenciones y separación de responsabilidades

- **[FUENTE]**: afirmación presente en una diapositiva; se indica PDF y página física.
- **[IMPLEMENTACIÓN PROPUESTA]**: diseño de tablas, controles y decisiones reproducibles para un proyecto predictivo; no es una regla obligatoria establecida en las diapositivas.
- **[VALIDAR CON GUÍA P2]**: decisiones dependientes del enunciado o de las columnas reales del dataset.
- Las diapositivas muestran ejemplos simulados (dos lunas, diez puntos, apartamentos, etc.). No trasladar sus números al conjunto hospitalario.
- El color **azul** y **naranja** suele separar clases o grupos en los ejemplos, pero su significado exacto depende de la leyenda de cada figura; **no** equivale automáticamente a supervivencia/muerte.
- Los recuadros **Key idea** resaltan conclusiones; **Intuition** interpreta conceptos; **Watch out** señala advertencias y límites; **Example/Your turn** son ejercicios, no requisitos del proyecto.

## 2. Reglas de negocio y controles para predicción de mortalidad hospitalaria

**[IMPLEMENTACIÓN PROPUESTA — validar con P2 y diccionario de datos]**

| ID | Regla o control | Implementación / verificación |
|---|---|---|
| RB-01 | Definir unidad de análisis | Una fila por observación elegible; decidir si es paciente o ingreso y documentar repetición de pacientes. |
| RB-02 | Variable objetivo | `hospdead` como candidato por el notebook; verificar definición oficial, codificación, nulos y temporalidad. No asumir que 1 significa fallecimiento sin validación. |
| RB-03 | Evitar fuga de información | Excluir variables registradas después del instante de predicción, incluyendo resultados de la hospitalización si el objetivo se predice al ingreso. |
| RB-04 | Partición antes de aprender transformaciones | Separar train/validación/test antes de imputar, escalar, seleccionar variables o ajustar PCA. |
| RB-05 | Pacientes repetidos | Si hay identificador de paciente y múltiples registros, usar partición por grupo para evitar contaminación entre conjuntos. |
| RB-06 | Valores faltantes | Medir tasa por columna y por partición; imputadores ajustados únicamente en entrenamiento; preservar indicadores de ausencia cuando proceda. |
| RB-07 | Variables categóricas | Codificación ajustada en train; tratar categorías desconocidas en validación/test. |
| RB-08 | Desbalance | Informar prevalencia; complementar accuracy con recall, precision, F1, matriz de confusión, ROC-AUC y PR-AUC cuando aplique. |
| RB-09 | Umbral | Escoger umbral usando validación y una función de costo clínico explícita; no optimizar sobre test. |
| RB-10 | Trazabilidad | Guardar semilla, split, parámetros, versión de código, métricas, columnas y transformaciones. |
| RB-11 | Interpretación clínica | No presentar asociaciones predictivas como relaciones causales ni como recomendación clínica autónoma. |

## 3. Fórmulas transversales y definiciones operativas

### Clasificación binaria

- Puntaje lineal: $z=\mathbf w^T\mathbf x+b$.
- Sigmoide: $\sigma(z)=1/(1+e^{-z})$; probabilidad estimada $p=P(y=1\mid x)$.
- Regla con umbral $\tau$: $\hat y=\mathbf1[p\geq\tau]$.
- Log-loss: $-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$.
- Matriz de confusión: TP, FP, TN, FN, siempre con clase positiva explícita.
- Accuracy $=(TP+TN)/N$; precision $=TP/(TP+FP)$; recall/sensibilidad $=TP/(TP+FN)$; especificidad $=TN/(TN+FP)$; $F_1=2PR/(P+R)$.
- Ridge $\lambda\|w\|_2^2$; lasso $\lambda\|w\|_1$; verificar convención de factor $1/2$ o $1/n$ del algoritmo.

### Árboles y ensambles

- Gini $G(m)=1-\sum_k\hat p_{mk}^2$; entropía $H(m)=-\sum_k\hat p_{mk}\log_2\hat p_{mk}$.
- Ganancia de un split: $\Delta I=I(m)-\frac{n_L}{n_m}I(L)-\frac{n_R}{n_m}I(R)$.
- Árbol de clasificación: predicción por mayoría de clase en hoja; probabilidad como proporción de clases de la hoja.
- Bagging: promedio de predicciones para regresión y voto/promedio de probabilidades para clasificación; bootstrap de filas.
- Random forest: bagging de árboles con submuestreo aleatorio de variables candidatas en splits.
- Boosting: modelos secuenciales que corrigen errores/residuos; distinguir AdaBoost de gradient boosting y sus funciones de pérdida.

### Kernels, agrupamiento y reducción

- Kernel: $K(x,x')=\phi(x)^T\phi(x')$; matriz Gram $K_{ij}=K(x_i,x_j)$, simétrica y semidefinida positiva.
- Lineal $x^Tx'$; polinomial $(c+x^Tx')^d$; RBF $\exp(-\gamma\|x-x'\|^2)$.
- K-means: $\min_{C,\mu}\sum_i\|x_i-\mu_{c_i}\|^2$; asignar al centro más cercano y recalcular medias.
- PCA: centrar $X$, diagonalizar matriz de covarianzas; varianza explicada $\lambda_j/\sum_k\lambda_k$; transformar con componentes aprendidas solo en train.
- Neurona: $z=\sum_j w_jx_j+b$, $a=\phi(z)$; red multicapa compone transformaciones y activaciones no lineales.

## 4. Esquemas lógicos de tablas (propuesta; no extraídos literalmente de las diapositivas)

### `dataset_observation`

| Columna | Tipo lógico | Restricción / significado |
|---|---|---|
| `observation_id` | STRING | Identificador único de fila generado de forma reproducible. |
| `patient_id` | STRING NULL | Solo si el origen proporciona identificador válido; útil para partición por grupos. |
| `prediction_time` | DATETIME NULL | Momento en que estarían disponibles las variables. |
| `hospdead` | INTEGER NULL | Objetivo candidato; verificar clases y codificación. |
| `feature_*` | NUMERIC/CATEGORICAL | Columnas reales del CSV, sin inventar nombres. |
| `split` | ENUM | train, validation, test; asignación inmutable. |

### `feature_dictionary`

| Columna | Tipo | Significado |
|---|---|---|
| `feature_name` | STRING | Nombre original. |
| `dtype_source` | STRING | Tipo leído desde CSV. |
| `semantic_type` | ENUM | numérica, ordinal, nominal, temporal, identificador, objetivo. |
| `missing_pct` | FLOAT | Fracción de nulos, por split. |
| `available_at_prediction` | BOOLEAN | Control anti-leakage. |
| `transformer` | STRING | Imputación, encoding, escalado o exclusión. |

### `model_experiment`

| Columna | Tipo | Significado |
|---|---|---|
| `experiment_id` | STRING | Clave del experimento. |
| `model_family` | STRING | LogisticRegression, Tree, RF, Boosting, SVM, etc. |
| `params_json` | JSON | Hiperparámetros y semilla. |
| `split_hash` | STRING | Identidad de particiones. |
| `pipeline_version` | STRING | Versión de transformaciones. |
| `selection_metric` | STRING | Métrica usada para elegir el modelo. |

### `prediction_result`

| Columna | Tipo | Significado |
|---|---|---|
| `experiment_id` | STRING | Referencia a experimento. |
| `observation_id` | STRING | Referencia a observación. |
| `split` | STRING | Conjunto evaluado. |
| `y_true` | INTEGER | Etiqueta real. |
| `score_positive` | FLOAT | Probabilidad o score; no confundir con clase. |
| `threshold` | FLOAT NULL | Umbral aplicado si corresponde. |
| `y_pred` | INTEGER | Predicción binaria. |

### `evaluation_metric`

| Columna | Tipo | Significado |
|---|---|---|
| `experiment_id` | STRING | Modelo evaluado. |
| `split` | STRING | Validation o test. |
| `metric_name` | STRING | Recall, precision, ROC-AUC, etc. |
| `metric_value` | FLOAT | Resultado. |
| `positive_label` | STRING | Definición de clase positiva. |
| `threshold` | FLOAT NULL | Relevante para métricas basadas en etiquetas. |

## 5. Flujo de transformaciones recomendado

```text
CSV original → perfilado de columnas y calidad → definición de unidad/objetivo
    → exclusión de fuga y duplicados según llave de negocio
    → partición reproducible train/valid/test (agrupada si procede)
    → [fit solo en train] imputación + encoding + escalado cuando corresponda
    → selección de variables / PCA opcional dentro del pipeline
    → entrenamiento y búsqueda de hiperparámetros en train/CV
    → selección y umbral en validación → evaluación final única en test
    → exportación de predicciones, métricas y metadatos
```

**Escalado:** necesario o especialmente relevante para regresión logística regularizada, SVM RBF, k-means, PCA y redes neuronales; los árboles y bosques normalmente no lo requieren. **PCA/clustering:** no deben introducirse automáticamente como requisitos de P2 por el solo hecho de aparecer en las clases.

## 6. Análisis detallado de las nueve presentaciones, por diapositiva

A continuación se conserva el contenido técnico textual de **cada página**, con su referencia exacta y una guía de lectura de los elementos visuales. La extracción textual no sustituye inspección manual de todos los gráficos; las interpretaciones de colores que no tengan leyenda explícita quedan marcadas como no verificadas.

### 1. ML_Course_Decision_Trees_Final.pdf — 63 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 08 — Decision Trees
Recursive partitioning, impurity, and pruning
— the interpretable model
Jan Hurtado
UTEC
September 15, 2026
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Roadmap
1 A model you can read
2 Fundamentals: what should a leaf predict?
3 Impurity: scoring a single node
4 Scoring a split: information gain
5 Finding the best split
6 Growing the tree: CART
7 Regression trees
8 Complexity and overfitting
9 Pruning
10 Reading the tree: interpretability
11 Limits, and the bridge to ensembles
12 Practice and wrap-up
Jan Hurtado (UTEC) Decision Trees September 15, 2026 2 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where we are in the course
Sessions 02--07: one weighted score Session 08: nested yes/no questions
Sessions 02–07 all fitted weights to a (possibly (no straight line can do better than 9/10) (4 boxes: 10/10)
kernelized) linear score f (x) = wTϕ(x) + b. 10 10
8 8
Everything changed only in the loss and the x2 6 x2 6
regularizer: squared error, log-loss, hinge; ridge, 4 4
lasso, the RBF kernel. 2 2
Today we change the model family itself. No dot 0 0 2 4 6 8 10 0 0 2 4 6 8 10
possible line: 90% class 0 products, no gradients, no distances — and no bestlogistic x1 x1 regression: 70% class 1
feature scaling.
A decision tree asks a sequence of yes/no questions
and predicts a constant at the end of each path.
Intuition
This is the most interpretable model in the course: you can
read it aloud. It is also the building block of Session 09’s
random forests and boosting.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 3 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Key terms (for newcomers)
Decision tree: a flow chart of yes/no Information gain: the impurity drop a split
questions ending in a prediction. produces (what we maximize).
Root / internal node / leaf: the top Recursive partitioning: splitting, then
question / a question / a final region. splitting each part, and so on.
Split: a test “xj ≤t?” that divides a node in Depth / number of leaves: how big the tree
two. is; this is its complexity.
Impurity: how mixed a node’s labels are (pure Pruning: cutting a grown tree back to avoid
= one class). overfitting.
Gini / entropy: the two standard impurity CART: the classic algorithm we derive today.
measures.
Key idea
Every one of these is defined and derived in the next 70 slides. Nothing is assumed.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 4 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Notation for this lecture
Ten points, two features, two classes
Data D = {(xi, yi)}ni=1, features x1, . . . , xp. class 0
10 class 1 (5,10)
A split is a pair (j, t): feature index j, threshold t. It (3,9)
sends x left iff xj ≤t.
8 (4,8)
Nodes are indexed by m. Node m owns nm training (10,7)
points and a box Bm ⊆Rp. (9,6)
6
ˆpmk = fraction of class k among the nm points of node x2 (6,5)
m. (1,4)
4
T is a tree, |T| its number of leaves, Tm the subtree (8,3)
hanging below node m. (7,2)
2
(2,1) the odd point
Watch out
Careful: t is always a threshold, never a node. Many text- 0
books use t for a node; we use m, so that R(m) and R(T) 0 2 4 x1 6 8 10
never clash.
The running example: ten points, two features,
two classes.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 5 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The one example we will carry all lecture
Ten points, two features, two classes
class 0
10 class 1 (5,10)
(3,9)
(4,8)
8
(10,7)
(9,6)
6
x2 (6,5)
(1,4)
4
(8,3)
(7,2)
2
(2,1) the odd point
0
0 2 4 6 8 10
x1
Ten points, integer coordinates, five per class. Small enough that every number on every slide can be
checked by hand — and we will check them.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 6 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Twenty questions, formalized
1 Start at the root with all n points.
2 Ask one question: “is xj ≤t?”
3 Send the point left or right. Ask another question. Repeat.
4 When you stop asking, you are at a leaf: predict its constant.
Key idea
The model is a function ˆf (x) = Pm∈leaves cm 1{x ∈Bm}: piecewise constant on a partition of
feature space. Two questions remain — what are the boxes Bm, and what is cm?
Intuition
Everything else today is answering those two questions, in the opposite order: first the constant
(easy), then the boxes (the real work).
Jan Hurtado (UTEC) Decision Trees September 15, 2026 7 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why a leaf is a box
A path is an AND of inequalities, so a leaf is a box: RF = x : x1 > 3.5 ∧2.5 < x2 5.5 x1 > 3.5 and x2 > 2.5 © ªand x2 5.5
10
8
6
x2 x2 x2
4 RF
2
0
0 2 4 6 8 10 0 2 4 6 8 10 0 2 4 6 8 10
x1 x1 x1
A path from root to leaf is an AND of inequalities. Each inequality is a half-plane; intersecting
half-planes gives an axis-aligned box. That single fact explains the whole geometry of trees.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 8 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The finished tree, as regions
The four leaves, as four boxes
Four leaves, four boxes, ten points, 10/10
10 correct.
Leaf A: x1 ≤3.5 — 3 points, all class 0.
RC
n = 4, [0, 4]
8 class 1 Leaf C: x1 > 3.5, x2 > 5.5 — 4 points, all
class 1.
6 RA Leaf F: x1 > 3.5, 2.5 < x2 ≤5.5 — 2
x2 n = 3, [3, 0]
class 0 points, class 0.
RF
4 n = 2, [2, 0] Leaf E: x1 > 3.5, x2 ≤2.5 — 1 point, class
class 0
1.
2 RE Intuition
n = 1, [0, 1] No straight line separates these ten points. Four class 1
boxes do it exactly.
0
0 2 4 6 8 10
x1
Jan Hurtado (UTEC) Decision Trees September 15, 2026 9 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The finished tree, as a flow chart
root: x1 ≤3.5?
n = 10, [5, 5], Gini = 0.5
yes no
leaf A
node B: x2 ≤5.5?
n = 3, [3, 0]
n = 7, [2, 5], Gini = 0.408
predict 0
yes no
leaf C
node D: x2 ≤2.5?
n = 4, [0, 4]
n = 3, [2, 1], Gini = 0.444
yes no predict 1
leaf E leaf F
n = 1, [0, 1] n = 2, [2, 0]
predict 1 predict 0
[n0, n1] is the class count. This is the same model as the previous slide — boxes and flow chart are two views of
one object. We refer to nodes A–F by these names for the rest of the lecture.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 10 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
First sub-question: the constant in a leaf
A leaf sees only its own points. So the question is small and self-contained:
Key idea
Given labels y1, . . . , ynm in a node, which single constant c minimizes the total loss Pi ℓ(yi, c)?
The answer depends only on the loss, and we already know all three answers:
0/1 loss ⇒the majority class.
squared loss ⇒the mean.
log-loss, predicting a probability ⇒the observed frequency ˆp.
Intuition
Let us derive each one, because the value of the minimum is exactly what impurity will turn out to be.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 11 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Restrict, then count
Restrict to the node, then count: ˆpk = nk/n
12 class 0
ˆp = (5/10, 5/10) class 1
= (0.500, 0.500)
10
8 ˆp = (2/7, 5/7) points 5 = (0.286, 0.714)
of
6
5 number 4
5
2
2
0
root node B
(all 10 points) (x1 > 3.5, 7 points)
Everything a node knows is a count vector. Root: [5, 5], so ˆp = (0.5, 0.5). Node B: [2, 5], so
ˆp = (2/7, 5/7) ≈(0.286, 0.714). A tree never looks at where the points are inside the box — only how
many of each class.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 12 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Best constant under 0/1 loss: the majority
Predict class k for all nm points. The number of Node B is [2, 5]: the majority class wins
mistakes is
0.8
0.7143
X 1{yi ̸= k} = nm (1 −ˆpmk). node
0.6 i the minc err(c)=1−maxk ˆpk = 2/7 in
This is smallest exactly when ˆpmk is largest: error 0.4
0.2857
cm = arg max ˆpmk, error = 1 −max ˆpmk. k k training 0.2
0.0
guess class 0 guess class 1
Example: Node B
[2, 5]: predicting class 1 costs 2 mistakes, predicting class
0 costs 5. Majority wins: error = 1−5/7 = 2/7 ≈0.286.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 13 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Best constant under squared loss: the mean
absolute loss: flatMinimize J(c) = Pi(yi −c)2. Differentiate: 900 Squared loss: the best constant is the mean minimum on [5,
8 10] 138 + 8(7.5 −c)2
800
6
4 700J′(c) = −2 X(yi −c) = 0 =⇒c = 1 X yi = ¯ym.
nm
0 i i −c)2 600 2 0 5 10 15
(yi 500
J′′(c) = 2nm > 0, so this is the unique minimum. The X= i 400 minimum at c = ¯y = 7.5
300 J = 138minimum value is Pi(yi −¯ym)2 — the within-node J(c)
sum of squares, which becomes our regression 200
J ′(c) = −2 (yi −c) = 0 ⇒c = ¯y i Ximpurity. 100
0
Watch out 0 2 constant4 prediction6 c (rent,8 hundreds10 of EUR)12 14
Change the loss and you change the constant: under
absolute loss the best constant is the median, and the
curve has a flat minimum (inset).
Jan Hurtado (UTEC) Decision Trees September 15, 2026 14 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Best constant probability: the observed frequency
Node B (ˆp = 5/7): both losses are minimized at q = ˆp --- and the minimum IS the impurity
4.0 log-loss −[plog2 q + (1 −p)log2 (1 −q)] 1.4 Brier 2[p(1 −q)2 + (1 −p)q2]
3.5 1.2
entropy H = 0.8631 bits Gini G = 0.4082
node 3.0 1.0
the 2.5
in 0.8
loss 2.0 0.6
1.5
0.4 average 1.0
0.5 0.2
0.0 0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
constant prediction q = ˆP(y = 1) constant prediction q = ˆP(y = 1)
Predict a probability q for class 1 in node B (ˆp = 5/7). Both standard scores — log-loss
−[ˆp log q + (1 −ˆp) log(1 −q)] and the Brier score (ˆp −q)2-style squared error — are minimized at q = ˆp.
Key idea
And their minimum values are the two impurity measures: log-loss at q = ˆp gives entropy; squared loss at
q = ˆp gives Gini. Impurity is not an arbitrary formula — it is “the cost of the best constant”.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 15 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
We need a number for “how mixed”
A leaf that is pure (one class) predicts perfectly. A leaf that is 50/50 is useless.
So define I(m) ≥0, the impurity of node m, with I = 0 iff the node is pure, and I maximal when
classes are balanced.
We will meet three candidates — and we will find that two of them work and one silently fails.
Key idea
All three are “the cost of the best constant” from the previous section:
error = 1 −max ˆpk, Gini = 1 − X ˆp2k, H = − X ˆpk log2 ˆpk.
k
k k
Jan Hurtado (UTEC) Decision Trees September 15, 2026 16 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where entropy comes from: surprise
Rare events are surprising; certain ones are not
How surprising is an event of probability p?
We want s(1) = 0 and s decreasing. 5
Independent events should add surprise: (bits) 4
s(pq) = s(p) + s(q). The only continuous p 3 p = 0.125: 3 bits
solution is s(p) = −log p. −log2
p = 0.25: 2 bits
2 1 bit = one fair yes/no answer In bits (log2): a certainty costs 0 bits, a coin surprise p = 0.5: 1 bit
flip 1 bit, a 1/4 event 2 bits. 1
Entropy is average surprise: 0 p = 1: 0 bits
H = Pk ˆpk (−log2 ˆpk). 0.0 0.2 probability0.4 p of what0.6happened 0.8 1.0
Intuition
Entropy = the average number of yes/no questions
needed to identify the class of a random point in the
node. Pure node: zero questions.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 17 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where Gini comes from: a disagreement probability
Node B: draw a point, draw a label independently
label drawn label drawn
is class 0 is class 1
point 4/49 10/49
is class 0 (agree) (disagree)
point 10/49 25/49
is class 1 (disagree) (agree)
P(disagree) = 1049 + 1049 = 2049 = 0.408 = G
Draw a point from the node, then draw a label independently from the node’s own distribution. Gini is
exactly the probability they disagree: Pk ˆpk(1 −ˆpk) = 1 −Pk ˆp2k.
Example: Node B, ˆp = (2/7, 5/7)
Gini = 1 −(2/7)2 −(5/7)2 = 1 −449 −2549 = 2049 ≈0.408.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 18 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The three curves, side by side
zoom: the error rate is a straight
all three vanish at the pure ends and peak at p = 1/2 line here, the others curve
0.50
0.5 1.0
0.45
0.4 0.8 0.40
error 0.3 0.6 (bits) 0.35
/ 0.30
Gini 0.2 0.4 entropy 0.25
0.20
0.1 0.2 0.15
0.10 (entropy halved to share the axis)
0.0 0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90
class-1 fraction p in the node p
Gini 2p(1 −p) error min(p, 1 −p) entropy (bits)
All three vanish at p = 0 and p = 1 and peak at p = 1/2. The zoom is the important part: the error
rate is a straight line on each half, while Gini and entropy curve. That difference decides which ones
can grow a tree.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 19 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A sanity check with more than two classes
A 3-class root with Gini 0.667 is at its maximum, not broken
With K classes, the worst case is ˆpk = 1/K
Gini 1 −1/K
node 2.0 entropyerror 1 −1/Klog2 K (bits) 2.000 for all k.
1.585 Then Gini = 1 −K · K2 = 1 −1K andmixed 1.5 1
H = log2 K.
evenly 1.000
an 1.0 K = 2: Ginimax = 0.5, Hmax = 1 bit.
of 0.667 0.667 0.750 0.750
0.500 0.500 K = 3: Ginimax = 0.667, Hmax = 1.585 bits.
0.5impurity Watch out
0.0 K = 2 classes K = 3 classes K = 4 classes So a Gini of 0.667 is not a bug — with three
classes it is the maximum. Always compare impu-
rity against 1 −1/K, not against 0.5.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 20 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Impurity in our running example
Impurity falls as we go down the tree
the three internal nodes the same nine numbers
root: G=0.500 G Gini H=1.000 10 entropy H (bits) E=0.500 1.000
1.0 error E
B: G=0.408 0.918
H=0.863 0.863
8 E=0.286
0.8
6
x2 0.6
0.500 0.500
0.444 D: G=0.444 impurity
0.408 4 H=0.918 E=0.333 0.4 0.333
0.286
2 0.2
0 0.0
0 2 4 6 8 10 root node B node D
x1
The three internal nodes scored by all three measures. Impurity falls as we descend: root 0.500 →node
B 0.408 →node D 0.444 — but note node D is not lower than B. Only the size-weighted total is
guaranteed to fall, which is the subject of the next section.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 21 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
From one node to two children
A candidate split (j, t) sends nL points left and nR right, nm = nL + nR.
Key idea
Information gain = parent impurity −weighted child impurity:
∆I(j, t) = I(m) −nL I(L) −nR I(R).
nm nm
The weights nL/nm and nR/nm are not cosmetic — without them the criterion is broken.
We maximize ∆I over every feature j and every threshold t. That is one node’s work.
Intuition
Read it as: “how many bits (or how much Gini) did asking this question buy me?”
Jan Hurtado (UTEC) Decision Trees September 15, 2026 22 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why the children are size-weighted
The children count in proportion to their size: 100(0.2659)38 + 100(0.2248)62
parent (n=100)
0.5 0.4800 'free' yes (n=38) 'free' no (n=62)
0.4
gain
= 0.2396
0.3 weighted child impurity impurity = 0.2404
Gini 0.2659
0.2 0.2248
0.1
0.0
parent children (width ∝n)
The spam split from Session 05: parent [60, 40], Gini = 0.48. Bar width is the sample count, so the
shaded area is “impurity mass”. Unweighted, a tiny pure leaf could outvote a huge mixed one; weighted,
the gain is an honest 0.48 −0.2404 = 0.2396.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 23 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 24

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The fundamental fact: concavity makes the gain non-negative
The gain is the gap from the curve down to the chord --- so a concave impurity gives gain
the root (unequal weights:Gini atparent is NOTsplit the midpoint) entropy, same split (botherrorchildren rate,≥0onnodeONE B'sstraightcut x2 segment)6.5 The parent’s class fraction is a mixture of the
parent parent
1.0 0.5 0.5
pL children’s: ˆp = + nL ˆpL nR ˆpR. pR 0.8 0.4 0.4 gain= 0.3958 nm nm gain= 0.2143 pR
= 0.0000 So the weighted child impurity is the height of the impurity 0.30.2 0.60.4 0.30.2 parentgain
chord between (ˆpL, I(ˆpL)) and (ˆpR, I(ˆpR)), 0.1 0.2 0.1
evaluated at ˆp. 0.00.0pL 0.2 class-10.4 fraction0.6 p 0.8 1.0 0.00.0pL 0.2 class-10.4 fraction0.6 p 0.8 1.0 0.00.0 0.2 class-10.4 fraction0.6 p 0.8 1.0pR
If I is concave, the curve lies above every chord
(Jensen’s inequality). Hence
∆I = I(ˆp) −chord(ˆp) ≥0.
Key idea
A split can never increase impurity. Gini and entropy are
strictly concave, so the gain is strictly positive whenever
ˆpL ̸= ˆpR.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 24 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 25

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What zero gain looks like
pL = pR = p: the gain is exactly 0 under Gini, entropy AND the error rate
the cut x2 2.5 at the root identical proportions ⇒ nothing was learned
1.2
10
p = 0.5 p = 0.5 p = 0.5
1.0
8
0.8
6
x2 0.6 proportions
4 class 0.4
2 0.2
0 0.0
0 2 4 6 8 10 parent left right
x1
Cut the ten points at x2 ≤2.5: children [1, 1] and [4, 4]. Both have exactly the parent’s class
proportions, ˆpL = ˆpR = ˆp, so the chord touches the curve. The gain is 0 under Gini, entropy and the
error rate. The split separated points but learned nothing.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 25 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 26

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why we do not grow with the error rate
A split that hands us a PURE leaf --- and the error rate scores it exactly zero
node B, cut x2 6.5: children [2, 2] and [0, 3] error: the chord lies ON the segment
Gini 0.1224 entropy 0.2917 error 0.0000 (slope −1 --- linear, not flat) Gini: the curve bends away, so the gap opens
10
0.5 0.5
pL pL parent
8
0.4 0.4
gain
= 0.1224
6 0.3 parentgain 0.3 x2
= 0.0000
4 0.2 0.2
2 0.1 0.1
0 0.0 0.0
0 2 4 6 8 10 0.0 0.2 0.4 0.6 0.8 1.0pR 0.0 0.2 0.4 0.6 0.8 1.0pR
x1 class-1 fraction p class-1 fraction p
Node B cut at x2 ≤6.5 hands us a pure leaf [0, 3] — obviously progress. Gini scores 6/49 ≈0.122 and
entropy 0.191, but the error rate scores exactly 0: both children keep majority class 1, and on that
straight segment the chord lies on the curve.
Watch out
A greedy grower using the error rate would reject this split and stop. Grow with Gini or entropy; the error
rate is fine for evaluating a finished tree.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 26 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 27

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Gini vs. entropy: does the choice matter?
Gini and entropy prefer the left cut; the ERROR RATE cannot tell them apart
x1 3.5 children [3, 0] and [2, 5] x2 5.5 children [4, 1] and [1, 4]
Gini 0.2143 entropy 0.3958 Gini 0.1800 entropy 0.2781
10
8
6
x2 x2
4
2
0
0 2 4 6 8 10 0 2 4 6 8 10
x1 x1
error gain 0.3000 error gain 0.3000
The root’s two best candidate cuts. Gini and entropy agree: x1 ≤3.5 (gain 0.2143 / 0.2955) beats
x2 ≤5.5 (gain 0.1800 / 0.2564). The error rate ties them at 0.2 and cannot choose.
Intuition
In practice Gini and entropy pick the same split the overwhelming majority of the time. Gini is the default
because it avoids a logarithm. Do not agonize over it; do agonize over tree size.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 27 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 28

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Only n −1 thresholds can ever matter
Sort the feature, test the midpoints: 9 candidates for 10 points
t is continuous, so there are infinitely class 0 class 1
many candidate splits — but only finitely x1
many distinct partitions.
Two thresholds between the same pair of 0.25 0.2143
sorted values give the identical split. 0.20
∆I(t)
0.1250 0.1250 So: sort the feature, test the midpoints
of consecutive distinct values. For n decrease 0.150.10 0.0833 0.0833
0.0556 givesany tthein thissamebandsplit 0.0556 points that is at most n −1 candidates. Gini 0.05
0.0200 0.0238
The gain is therefore a step function of t, 0.00
not a smooth curve — there is nothing to 2 4 threshold t on6 x1 8 10
differentiate, so we enumerate.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 28 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 29

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Scanning both features
Scan every feature, keep the single best (j, t)
0.25 feature x1
feature x2
winner: x1 3.5, gain 3/14 = 0.2143
(runner-up 0.1800)
0.20
0.15 decrease
Gini 0.10
0.05
gain = 0 gain = 0
0.00
2 4 6 8 10
threshold t
Nine candidates on x1, nine on x2. The winner is x1 ≤3.5 with gain 3/14 = 0.2143; the runner-up is
x2 ≤5.5 with 0.1800. Two candidates on x2 score exactly zero (hollow circles).
Going deeper (optional)
Cost. Sorting each feature once costs O(pn log n); then each node scans its points in O(pnm) by updating
the counts incrementally as the threshold sweeps. Building a balanced tree costs about O(pn log n) overall.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 29 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 30

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A tree only sees the order of a feature
A split depends only on the ORDER of a feature's values --- monotone transforms are free, rotations are not
replace x1 by logx1: rotate the SAME data by 30 ◦:
the ten points and their tree same tree, threshold 1.24 6 leaves instead of 4
12
10 10
10
8 8
x2 8
6 6
x2 x2
6 rotated
4 4
4
2 2
2
0 0
0 2 4 6 8 10 0.0 0.5 1.0 1.5 2.0 2.5 2 0 2 4 6
x1 logx1 rotated x1
Replace x1 by log x1: the sorted order is unchanged, so the tree is identical (only the printed threshold
moves). This is why trees need no standardization — unlike every model in Sessions 02–07.
Watch out
But rotation is not a monotone transform of one feature. Rotating the same ten points by 30◦forces the
tree from 4 leaves to 8: the axes matter.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 30 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 31

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Categorical features
Categorical features: exponentially many subsets, but one sort makes the search linear
Sort the categories by their class-1 rate; then only the
k −1 = 4 contiguous cuts can win (and the best of all 15 is one of them) why the shortcut matters 2047
1.2 2k −1 all subsets: −1 split here:
k after sorting: 103 A, B C, D, E −1 0.914
n = 35 1.0 © ª vs © ª 0.800
n = 25 category 0.8 test 102 the 0.560 to
n = 50 in 0.6
rate splits
101
11 n = 30
class-1 0.40.2 0.100n = 40 0.267
100
0.0
A B C D E 2 4 6 8 10 12
category, sorted by class-1 rate number of categories k
A k-category feature could be split by any subset: 2k−1 −1 of them. Breiman’s shortcut: sort the
categories by their class-1 rate and test only the k −1 contiguous cuts — provably optimal for binary
classification. Here the best of all 15 subsets is {A, B}, one of the 4 contiguous cuts.
Watch out
scikit-learn does not implement categorical splits: encode first. And beware — a high-cardinality category
has many cuts to get lucky with.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 31 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 32

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The algorithm, in full
Key idea
BuildTree(node m with points S):
1 If a stopping rule fires, make m a leaf: cm ←majority class (or mean ¯y). Return.
2 For every feature j and every midpoint t of S’s j-values, compute ∆I(j, t).
3 Keep the best (j⋆, t⋆). If ∆I(j⋆, t⋆) is not an improvement, make m a leaf and return.
4 Split S into SL, SR; recurse on both.
Watch out
Greedy: step 3 commits to the best split now, never reconsidering. Finding the globally optimal tree
is NP-hard, so this is a heuristic — an excellent one in practice, with one famous failure mode we
will see in two slides.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 32 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 33

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Growing our example, one split at a time
Greedy growth: the newly added cut is orange
split 1: x1 3.5, gain 3/14 = 0.214 split 2: x2 5.5 at B, gain 32/147 = 0.218 split 3: x2 2.5 at D, gain 4/9 = 0.444
training accuracy 0.8 training accuracy 0.9 training accuracy 1.0
10
8
6
x2 x2 x2
4
2
0
0 2 4 6 8 10 0 2 4 6 8 10 0 2 4 6 8 10
x1 x1 x1
Split 1: x1 ≤3.5, gain 0.2143, accuracy 0.8. Split 2 (inside node B): x2 ≤5.5, gain 0.2177, accuracy
0.9. Split 3 (inside node D): x2 ≤2.5, gain 0.4444, accuracy 1.0. The newly added cut is orange.
Intuition
Note the gains do not decrease monotonically — the deepest split has the largest gain, because it acts on a
small, very mixed node.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 33 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 34

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Watch a tree grow (animated)
300 two-moons points. Each frame adds the highest-gain split. The early splits capture real structure;
the late ones carve slivers around single noisy points. (Animated PDF — press play.)
Jan Hurtado (UTEC) Decision Trees September 15, 2026 34 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 35

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
When do we stop?
Pre-pruning: four different knobs, three of them landing on the same 3-leaf tree
A node becomes a leaf when any of these fires: min_samples_leaf=2 min_impurity_decrease=0.15 max_depth=1= 2, train 0.8 max_depth=2= 3, train 0.9 0.9 |T| |T| |T| = 3, train |T| = 3, train 0.9
it is pure (all one class) — nothing left to gain;
it has too few points (min_samples_split,
min_samples_leaf);
a depth limit is reached (max_depth);
Four knobs on the ten points; three land on the
the best gain is below a floor same 3-leaf tree.
(min_impurity_decrease);
a leaf budget is exhausted (max_leaf_nodes).
These are the pre-pruning knobs. With none of them
set, a tree grows until every leaf is pure — training
accuracy 1.0, and heavy overfitting.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 35 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 36

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The failure mode of greedy + early stopping: XOR
Greedy search is myopic: a useless-looking split can be the one that makes the next split perfect
200 points: the depth-2 tree
the smallest XOR: four points x1 0 then x2 0 is 100% correct the root's gain curve: flat zero
2.0 0.0150 feature x1 1.25
y = 1 y = 0 1.5 0.0125 early-stopping threshold feature x2
1.00 1.0 ∆I(t) 0.0100 (min_impurity_decrease = 0.01)
0.75 0.5 0.0075
x2 0.50 x2 0.0 0.0050 0.25 0.5 decrease 0.0025
0.00 y = 0 y = 1 1.0 Gini 0.0000 every candidate sits on zero
0.25 1.5 0.0025 ⇒1 theleaf,tree50%neveraccuracysplits:
2.0 0.0050
0.0 0.5 1.0 2 1 0 1 2 1.5 1.0 0.5 0.0 0.5 1.0 1.5
x1 x1 threshold t
either cut gives children [1, 1] and [1, 1]
gain = 0 exactly
In XOR, every candidate split at the root has gain exactly 0 (here ≤5.6 × 10−17, i.e. floating-point
zero) — yet the depth-2 tree x1 ≤0 then x2 ≤0 is 100% correct. A useless-looking split is the one
that makes the next split perfect.
Watch out
So min_impurity_decrease= 0.01 stops at the root: 1 leaf, 50% accuracy. This is the argument for growing
first and pruning after.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 36 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 37

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The other structural limit: axis-aligned cuts
A tree can only cut parallel to the axes --- a diagonal boundary costs it a staircase, and never quite fits
depth 2: 4 leaves depth 4: 14 leaves depth 8: 20 leaves same data, rotated 45 ◦
test 86.9% test 94.8% test 95.4% depth 1: 99.9%
1.0 1.0 1.0 0.8
0.6
0.8 0.8 0.8
0.4
0.6 0.6 0.6 x2 0.2
x2 0.0
0.4 0.4 0.4 rotated 0.2
0.4
0.2 0.2 0.2
0.6
0.0 0.0 0.0 0.8
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.5 1.0 1.5
x1 x1 x1 rotated x1
A perfectly clean diagonal boundary costs a tree a staircase: depth 2 gets 86.9%, depth 4 94.8%, depth
8 95.4% — always approximating, never exact. Rotate the very same data 45◦and one split gets 99.9%.
Intuition
This is the tree’s inductive bias, exactly as the RBF kernel’s smoothness was the SVM’s. A model is defined
as much by what it cannot express as by what it can.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 37 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 38

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The same recursion, a new criterion
Target y ∈R. Leaf m predicts the constant cm = ¯ym (derived earlier).
Node impurity becomes the sum of squared errors SSE(m) = Pi∈m(yi −¯ym)2 — equivalently nm
times the node’s variance.
The split criterion is unchanged in spirit: choose (j, t) minimizing SSE(L) + SSE(R), i.e.
maximizing the drop from the parent.
Key idea
Classification and regression trees are one algorithm. Swap “majority class” for “mean” and “Gini”
for “SSE” — everything else, including pruning, is identical.
Intuition
A regression tree is a staircase: it cannot slope, only step.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 38 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 39

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Eight apartments: the running regression example
Eight apartments: a regression tree predicts the MEAN inside each box
one leaf: predict ¯y = 7.5 split at area 4.5: predict 3.5 and 11.5 depth 2: four leaves, SSE = 2
SSE = 138 SSE = 5 + 5 = 10, gain = 128 (dotted: depth 3, one tread per point, SSE = 0)
14
EUR) 12
of 10
8
6 (hundreds
rent 4
2
0
1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8
area (tens of m2) area (tens of m2) area (tens of m2)
Areas 1..8 (tens of m2), rents 2, 3, 4, 5, 10, 11, 12, 13 (hundreds of EUR). One leaf: predict ¯y = 7.5, SSE
= 138. Best single split at area ≤4.5: predict 3.5 and 11.5, SSE = 5 + 5 = 10, so the gain is 128.
Depth 2 reaches SSE = 2; depth 3 gives one tread per point, SSE = 0.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 39 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 40

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Scoring every threshold
Every threshold scored by the squared error it leaves behind
160 20.0
parent SSE = 138
140 17.5 units)
120 15.0
103.43 103.43
100 12.5 (variance SSER
+ 80 71.33 71.33 10.0 point
SSEL 60 7.5 per
40.80 40.80
40 5.0 same,
20 10.00 best split: area 4.5 2.5 the
gain = 138 −10 = 128 0 0.0
1 2 3 4 5 6 7 8
threshold t on area
The same step-function scan as for classification, now in squared-error units:
103.43, 71.33, 40.80, 10.00, 40.80, 71.33, 103.43. The minimum is the split. Note the symmetry — this
dataset is a clean two-group structure.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 40 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 41

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Depth, chosen honestly
A regression tree is a staircase: depth sets the number of treads
depth 1 --- too shallow depth 5 --- chosen by 5-fold CV depth 12 --- too deep outside the training range
CV MSE 0.279 CV MSE 0.089 CV MSE 0.100 the fit is flat forever
true function
1.5 tree fit 1.5 1.5 2
1.0 1.0 1.0
1
0.5 0.5 0.5
y 0.0 0.0 0.0 0
0.5 0.5 0.5
1
1.0 1.0 1.0
1.5 1.5 1.5 2
2.0 2.0 2.0
2 0 2 2 0 2 2 0 2 4 2 0 2 4
x x x x
80 noisy points. Depth 1 underfits (CV MSE 0.279); depth 5 is selected by 5-fold CV (CV MSE
0.089); depth 12 chases noise (0.100). The fourth panel is the warning: outside the training range a
tree’s prediction is flat forever — it cannot extrapolate.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 41 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 42

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Tree size is the complexity knob
Filled dots are training points, open circles are test points
A shallow tree has few boxes: it cannot train 0.906 CV4 leaves0.889 test 0.817 train 0.961 CV8 leaves0.939 test 0.908 train 1.000 22CVleaves0.906 test 0.883
express much, so high bias, underfitting.
A fully grown tree puts a box around (almost)
every training point: training error 0, and it
has memorized the noise.
|T|, the number of leaves, plays the role λ
The same moons data at 4, 8 and 22 leaves. played for ridge and C, γ played for the SVM.
Watch out
With default settings, DecisionTreeClassifier
grows until pure. A tree is overfitted by default
— this is the one model in the course where doing
nothing is actively wrong.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 42 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 43

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The complexity curve
More leaves always help the training set; CV knows better
1.00
0.95
CV picks 8 leaves
0.90 CV 0.939
accuracy 0.85
0.80
training accuracy
0.75 5-foldtest accuracyCV accuracy(shown(onforthehonesty,trainingneverset) used to choose)
2 4 6 8 10 12 14 16 18 20 22
number of leaves |T|
Training accuracy rises monotonically — more leaves can only fit the training set better.
Cross-validated accuracy peaks at 8 leaves (0.9389), then falls to 0.9056 fully grown.
Watch out
Plotted as accuracy the curve is an inverted U (a hump); plotted as error it is the familiar U of Session 03.
Same picture, flipped.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 43 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 44

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The same story on the animation’s data
More leaves: training keeps rising, test plateaus then dips
1.00
0.95 gap
0.117
0.90 best test accuracy
0.908 at 7 leaves accuracy 0.85
0.80
training accuracy
0.75 test accuracy
3 6 9 12 15 18 21
number of leaves
Every frame of the animation, scored. Test accuracy peaks at 0.908 with 7 leaves, then decays to 0.883
while training accuracy marches to 1.000 — a generalization gap of 0.117 bought by 15 extra leaves.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 44 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 45

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where the error goes: variance
20 bootstrap resamples, 20 trees: thin gray = each tree, thick green = their average
stumps (2 leaves): high bias, low variance fully grown: low bias, high variance
20 bootstrap resamples, 20 trees, drawn on top of each other. Stumps (left) are all nearly the same
wrong line: high bias, low variance. Fully grown trees (right) scatter wildly: low bias, high variance.
Key idea
Trees sit at the high-variance end of the trade-off. Remember this picture — it is the entire motivation for
Session 09.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 45 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 46

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two ways to keep a tree small
Pre-pruning (stop early) Post-pruning (grow, then cut)
max_depth, min_samples_leaf, Grow the tree fully, then collapse the subtrees
min_impurity_decrease, max_leaf_nodes. that do not earn their keep.
Cheap: the tree is never built. Sees the whole tree before deciding.
Short-sighted: it cannot see a split that only This is cost-complexity pruning, and it is
pays off two levels down — exactly the XOR what we derive next.
failure.
Key idea
The two are not exclusive: a common recipe is a loose max_depth for speed, then cost-complexity
pruning chosen by cross-validation.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 46 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 47

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The cost-complexity objective
Key idea
For a tree T with leaves |T|, define the penalized cost
Rα(T) = R(T) + α |T|, α ≥0.
R(T) = Pm ∈leaves nmn I(m) is the tree’s training risk — the size-weighted impurity of its leaves.
(This is scikit-learn’s convention, in Gini units.)
α is a price per leaf. α = 0: keep everything. α huge: collapse to the root.
This is the same regularized-objective pattern as ridge (loss + λ∥w∥2) — fit plus a penalty on
complexity.
Intuition
A leaf must buy its place: it is kept only if it reduces R(T) by more than α.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 47 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 48

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The weakest link
Consider collapsing the subtree Tm below internal node m into a single leaf. Risk rises by
R(m) −R(Tm) ≥0; leaves fall by |Tm| −1.
The collapse is worth it as soon as α exceeds the break-even price
R(m) −R(Tm)
g(m) = .
|Tm| −1
Weakest-link pruning: repeatedly collapse the node with the smallest g(m). This traces a finite,
nested sequence of subtrees T (0) ⊃T (1) ⊃· · · ⊃{root}.
Key idea
Although α is continuous, only finitely many trees can ever be optimal — one per interval between
consecutive breakpoints. So we never search a continuum: we search a short list.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 48 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 49

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The whole path, by hand, on the ten points
Cost-complexity pruning on the ten points: α buys leaves, and only four subtrees can ever win
4 leaves: R(T) = 0.0000 3 leaves: R(T) = 0.1333 2 leaves: R(T) = 0.2857 1 leaf: R(T) = 0.5000
collapsed at α = 0.1333 collapsed at α = 0.1524 collapsed at α = 0.2143
each subtree is a straight line; the winner is the lowest one the path is a staircase: finitely many trees
0.000 + 4α 4 leaves:
0.133 + 3α 3 leaves:
0.8
0.286 + 2α = 0.1333152 105=16 0.1524 = 0.2143143 2 leaves:
α|T| 1 leaf: 0.500 + 1α 4
+ 0.6 |T|
R(T) = kept 3
Cα(T) 0.4 leaves 2
cost 0.2
1
0.0
0.00 0.05 0.10 0.15 0.20 0.25 0.00 0.05 0.10 0.15 0.20 0.25
penalty per leaf α α
R(T) = 0, 152 , 27 , 12 for 4, 3, 2, 1 leaves. Each subtree’s cost is a straight line R(T) + α|T|; the winner is
whichever is lowest. They cross at α = 152 , 10516 , 143 — and scikit-learn returns exactly these numbers.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 49 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 50

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Worked: the first breakpoint
Collapse node D (the subtree E / F) into one leaf:
Before: leaves E and F are pure, so R(TD) = 0 and |TD| = 2.
After: node D holds [2, 1], Gini = 1 −(2/3)2 −(1/3)2 = 49, and it carries weight nD/n = 3/10. So
R(D) = 103 · 49 = 15.2
Break-even price:
R(D) −R(TD) 152 −0 2 g(D) = = = ≈0.1333.
|TD| −1 2 −1 15
Example: Reading it
For α < 2/15 the two pure leaves are worth their price and we keep them. At α = 2/15 we are indifferent.
Above it, they are collapsed — and the tree drops from 4 leaves to 3. The next two breakpoints, 16/105 and
3/14, are computed the same way.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 50 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 51

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Choosing α: cross-validation, not the test set
Sweep α, keep the subtree with the best CV accuracy
1.00 22 18 8 6 2
0.95
0.90
α = 0.00708
0.85 8 leaves
0.80 accuracy
0.75
0.70 training accuracy
5-fold CV accuracy
0.65 test at the chosen α (0.908)
test, fully grown (0.867)
0.60
0 10 4 10 3 10 2 10 1
cost-complexity penalty α (numbers on top: leaves |T|)
Score each of the 13 subtrees by 5-fold CV on the training set. Winner: α = 0.00708 →8 leaves, CV
0.9278, test 0.908 — versus 0.867 for the fully grown 22-leaf tree.
Watch out
The test markers are shown for honesty only — they played no part in choosing α. Selecting on the test set
is the Session 03 sin.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 51 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 52

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The one-standard-error rule
The CV curve is estimated from 5 folds, so it has a standard error — here SE = 0.0061 at the
optimum.
Differences smaller than one SE are noise. The 1-SE rule: among all α whose CV score is within
one SE of the best, choose the largest α — i.e. the smallest tree that is statistically
indistinguishable from the best.
You get a simpler, more stable, more interpretable model at no measurable cost.
Intuition
Same rule you met for ridge and lasso in Session 03. When two models tie, ship the smaller one.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 52 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 53

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The payoff: a model a clinician can read
A depth-3 tree on the breast-cancer data (398 training biopsies of 569): every prediction is a path you can read aloud
train 98.0% test 90.1% (8 leaves; grown out it takes 16 leaves to reach test 90.6%)
worst
perimeter <= 106.1
samples = 398
value = [148.0, 250.0]
class = benign
True False
worst worst
concave points <= 0.158 perimeter <= 115.35
samples = 241 samples = 157
value = [9, 232] value = [139, 18]
class = benign class = malignant
worst mean mean area error <= 48.975 concavity <= 1.046 symmetry <= 0.126 texture <= 19.48 samples = 235 samples = 35 samples = 6 samples = 122 value = [4, 231] value = [19, 16] value = [5, 1] value = [120, 2] class = benign class = malignant class = malignant class = malignant
samples = 234 samples = 1 samples = 5 samples = 1 samples = 16 samples = 19 samples = 1 samples = 121
value = [3, 231] value = [1, 0] value = [5, 0] value = [0, 1] value = [2, 14] value = [17, 2] value = [0, 1] value = [120, 1]
class = benign class = malignant class = malignant class = benign class = benign class = malignant class = benign class = malignant
class 0 = malignant, class 1 = benign; each box shows the split, the sample count and the class counts [n malignant, n benign]
Wisconsin breast cancer, 398 training biopsies. Depth 3, 8 leaves, test accuracy 90.1% — against 90.6% for the
fully grown 16-leaf tree. Half the leaves, the same accuracy, and every path is a sentence.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 53 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 54

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What “interpretable” actually buys you
The largest leaf — 234 of 398 patients — is the
single rule Watch out
Interpretable ̸= true. The tree is a summary of
worst perimeter ≤106.1 this sample. The thresholds (106.1, 0.158) are
estimated from data and, as the next section
and worst concave points ≤0.158 shows, they move when the data move.
and area error ≤48.98,
Intuition
which is 98.7% benign. Also note the depth-3 tree already has leaves
with a single patient. Small leaves are where
You can audit it: a clinician can say whether that
overfitting hides, even in a shallow tree.
rule is medically sensible.
You can explain a single prediction exactly — no
approximation, no surrogate model.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 54 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 55

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Trees give probabilities too
Shading and printed numbers are ˆP(y = 1) --- one value per leaf
stump: 2 leaves depth 2: 3 leaves full tree: 4 leaves
query (5, 4): ˆP(y = 1) = 0.714 query (5, 4): ˆP(y = 1) = 0.333 query (5, 4): ˆP(y = 1) = 0.000 1.0
10
0.8
8 1.000 1.000
leaf
0.6 the 6
x2 0.000 0.714 x2 0.000 x2 0.000 in
1)
4 0.000 0.4 =
ˆP(y
0.333
2 0.2
1.000
0 0.0
0 2 4 6 8 10 0 2 4 6 8 10 0 2 4 6 8 10
x1 x1 x1
ˆP(y = 1 | x) is just ˆp in the leaf that x lands in — one value per box, constant inside it. The same
query point gets a different probability from a small, a medium and a large tree.
Watch out
A deep tree’s leaves hold very few points, so its probabilities are extreme (0 or 1) and badly calibrated.
Shallow trees give smoother, more honest probabilities.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 55 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 56

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Feature importance: how it is computed
MDI = mean decrease in impurity: add up what each feature bought
the ROOT feature x1 is NOT the most important
root
x1 0.2143 → 0.4286
total = 0.5
= root impurity
normalized:
x1 = 3/7
x2 = 4/7
node B node
x2 0.1524 0.1333 D → 0.5714
0.00 0.05 0.10 0.15 0.20 0.25 0.30
summed weighted impurity decrease ntn ∆I(t)
MDI (mean decrease in impurity): for each feature, add up nmn ∆I(m) over the nodes that split on it.
Here x1 earns 0.2143 at the root; x2 earns 0.1524 + 0.1333 at nodes B and D. They sum to 0.5, the
root impurity, and normalize to 3/7 and 4/7.
Key idea
So the root feature is not the most important one. “Splits first” and “contributes most” are different
questions.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 56 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 57

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Feature importance: how it lies
A continuous feature gets ~499 thresholds to get lucky --- so MDI on a deep tree rewards it
MDI, fully grown tree
1.0 MDI, depth 3
0.895 permutation, held-out data
0.8 pure noise, ranked FIRST
0.6 0.602
importance 0.4 0.379
0.2 0.198
0.084
0.020 0.021 0.013 0.032
0.0
informative noise noise
(binary, 80%) (binary) (continuous)
Three features, 500 points: a binary feature agreeing with the label 80% of the time, binary noise, and
continuous noise. On a fully grown tree MDI ranks the pure noise first (0.602 vs 0.379).
Watch out
Why: a continuous feature offers ∼499 thresholds, so it can always find one that looks good on the training
data. Fixes: keep the tree shallow (MDI becomes 0.895 for the real feature), or use permutation importance
on held-out data.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 57 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 58

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The deal-breaker: instability
One point moved one unit --- and the tree asks a different first question
original data one point moved one unit right
first question: x1 3.5 (gain 0.2143) first question: x2 5.5 (gain 0.1800)
10
8
6
x2 x2
4
2
best x2 cut: 0.1800 best x1 cut: 0.1250
0
0 2 4 6 8 10 0 2 4 6 8 10
x1 x1
Move one point by one unit and the tree asks a different first question — so every region below it
changes. The model is not “slightly different”; it is a different flow chart with the same accuracy.
Intuition
The cause is the argmax in the split search: two candidates can be separated by 0.0001, and a tiny data
change flips which one wins. All the structure below then changes discontinuously.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 58 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 59

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Instability on real data
Resampleresamplethe#0:datarootandsplitstheontree'worstchangesperimeter'--- high variance is the tree's one realresampleweakness#2: root(andsplitssessionon 'worst09'sconcaveopening)points'
test 91.2% test 90.1% A bootstrap resample draws n
worst
perimeter <= concave pointsworst
samples samples
value = 398110.05225.0] value class= [173.0, = benign class= = points from the training set with True False True [155,benign= <=398243]0.12 False
pointsworst
samples = <=16718]0.06 replacement — a plausible concavevalue class= = 1630.145] valueareasamplesclass<=mean== [6,benign742.05 = 225]231 concaveclassvaluesamples=pointsmean=malignant[149, [15.0,= benign= <=235220.0]0.16 symmetryclassvaluesamples=mean=malignant[158,<=
alternative sample from the same valuesamplesclass == [5,benign = 220]225 classvaluesamples= =malignant[10,= 100] classvaluesamples==benign[0,= 3]3 classvaluesamples==malignant[158,= 1602] valuesamplesclass == [0,benign= 222]222 classvaluesamples= malignant= [6,= 3]9 valueclasssamples== [10,benign= 2616] classvaluesamples==malignant[139,= 1412]
source.
the trees differ, but they are all about equally good
Over 50 resamples of the cancer worstworstconcaveperimeterpoints 12 24 108 mean 91.6%
6 points count 64 5 radius data, the root split uses 5 different mean concaveworst
2 3 worst area
0 5 10 15 20 25 0 0.87 0.88 0.89 0.90 0.91 0.92 0.93 0.94 0.95 features.
times chosen as the ROOT split, over 50 bootstrap resamples test accuracy of each resampled tree
Yet test accuracy barely moves:
mean 91.6%, range 87.1–94.7%.
Key idea
Many different trees, all about equally
good. That is the signature of variance,
not bias — and variance is the one thing
averaging can remove.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 59 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 60

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Strengths and limits, honestly
Strengths Limits
Interpretable: read it aloud as rules. High variance — the deal-breaker.
No scaling, no centering — only order Axis-aligned: diagonals cost a staircase.
matters. Greedy: not globally optimal (NP-hard).
Mixed feature types; missing values handled Cannot extrapolate in regression.
by surrogate splits.
Biased importances; poorly calibrated
Captures interactions automatically (a path is probabilities when deep.
an AND).
Struggles with smooth, additive relationships
Fast prediction: one comparison per level. that a linear model nails.
Multi-class for free; no one-vs-rest needed.
Watch out
A single tree is rarely the best predictor. It is often the best explanation.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 60 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 61

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The bridge to Session 09
Averaging many high-variance trees smooths the boundary --- that is Session 09
one fully grown tree 20 trees on 20 bootstrap resamples their AVERAGE
test 0.867 (each one jagged, all different) test 0.900
One fully grown tree scores 0.867 on the test set. Twenty trees, each grown on a bootstrap resample, are
individually jagged and all different — but their average scores 0.900 with a visibly smoother boundary.
Key idea
Averaging B roughly independent estimators cuts variance by about B while leaving bias alone. Trees are
low-bias/high-variance, so they are the ideal thing to average. That is bagging, random forests and boosting.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 61 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 62

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Using trees in practice
DecisionTreeClassifier(criterion="gini")
— entropy rarely changes anything. Watch out
Common pitfalls
Never accept the defaults. Set max_depth or
Choosing depth on the test set.
ccp_alpha; the default grows to purity.
Reporting training accuracy (always 1.0
Tune ccp_alpha with for a grown tree).
cost_complexity_pruning_path +
Trusting MDI on high-cardinality
GridSearchCV. features.
Set random_state: ties in the split search are Expecting extrapolation in regression.
broken at random, so runs differ.
Re-running and being surprised the tree
Use class_weight="balanced" for imbalanced changed.
data.
Read importances with suspicion; prefer
permutation importance on a held-out set.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 62 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


#### Página física 63

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Summary
Key idea
A tree asks “xj ≤t?” at each node, carving feature space into axis-aligned boxes with one constant
per box. A leaf’s constant is the best constant under the loss: majority class, mean, or ˆp.
Impurity is the cost of that best constant: Gini = 1 −Pk ˆp2k, H = −Pk ˆpk log2 ˆpk. A split is scored
by its information gain ∆I = I(m) −nL I(L) −nR I(R), which is ≥0 because impurity is concave.
nm nm
The error rate is linear, so it can score a useful split at zero — grow with Gini or entropy.
CART is greedy: sort each feature, test the n −1 midpoints, keep the best (j, t), recurse. Greedy fails
on XOR, which is why we grow first and prune after.
Tree size is the complexity knob. Cost-complexity pruning Rα(T) = R(T) + α|T| traces finitely
many nested subtrees; pick α by cross-validation.
Trees are interpretable and scale-free but high variance. Averaging many of them is Session 09.
Jan Hurtado (UTEC) Decision Trees September 15, 2026 63 / 65
```

**Visual:** los rectángulos/regiones coloreados son hojas del árbol; las líneas verticales/horizontales representan umbrales de partición. En diagramas de flujo, flecha «yes» lleva a la rama que cumple `≤`; «no» a la complementaria. Colores distinguen clases según leyenda, no resultados clínicos.


### 2. ML_Course_Ensemble_Methods_Final.pdf — 54 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 09 —
Ensemble Methods
The intuitive version: many simple models, one
strong prediction
Jan Hurtado
UTEC
October 11, 2026
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Let’s start with a guess
Example: your turn (1 minute)
Everybody writes down a guess — the number of candies
in a jar, the height of this room, the year a building was
built. No talking. Then we average a dozen of your
guesses and compare.
In 1907, Francis Galton collected 787 guesses of the weight
of an ox at a country fair. A typical guess was off by about
3%. The middle guess, 1,207 lb, was within 1% of the
true 1,198 lb. (Galton, 1907; quartiles 1,162 and 1,236 lb)
how many?
Key idea
The crowd’s guess was about four times closer than a typical
single guess.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 2 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why does the crowd win? data: a simulated crowd
independent guesses everybody underestimates
truth 850 truth 850
30 average 860 average 733
20
people 10
0
250 500 750 1000 1250 250 500 750 1000 1250
guessed number of candies guessed number of candies
Left: 200 simulated people guess independently; a Right: the same people, but everybody also
typical person misses by 17%. Over 1,000 such underestimates. Now the average (733) is 14%
crowds the average misses by only 1.2%, and it off: an error everybody shares does not cancel.
beats 96% of the people in its crowd.
Key idea
Independent errors cancel when you average. Shared errors do not. That sentence is the whole
first half of today.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 3 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A reminder from session 08: a decision tree data: the ten points
1: x1 ≤3.5?
yes no
1 circle 2: x2 ≤5.5?
yes no
3: x2 ≤2.5? square
yes no
x2 2 square circle
A tree asks yes/no questions, one feature at a
3 time.
Each question cuts a box in two; each final box (a
leaf) gets one answer.
x1
A tree with one question is a stump. Remember
it: it comes back in boosting.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 4 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The problem: one tree is unstable data: the two moons
resample 1: test 0.867 resample 2: test 0.858 resample 3: test 0.858
Same recipe, same kind of data: three random resamples of the 180 training points
(slightly different copies of the data; we will see how). Three very different trees (19, 10
and 14 leaves).
Their test accuracy is about the same (0.858 to 0.867), yet their boundaries disagree: one
tree is jumpy.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 5 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Diagnosis: bias and variance, as darts
low bias low bias Each dart is one model, trained on a different
low variance HIGH variance dataset; the bull’s-eye is the truth.
Bias: the darts land off-center on average.
The model is too simple to hit the target.
Variance: the darts are scattered.
The model changes a lot with the training a deep tree
data.
HIGH bias HIGH bias
low variance HIGH variance A deep tree: low bias, high variance.
A stump: high bias, low variance.
Intuition
The unstable trees of the last slide are the top-right
board: right on average, scattered every time. Can
a stump we tighten the cluster?
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 6 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Today’s idea: a team of models
Key idea
An ensemble is a set of models whose predictions are combined (by voting or averaging)
into one prediction.
in parallel: a committee in sequence: a chain of tutors
all at once: bagging, random forests one after the other: boosting
model 1
model 2 vote / weighted
data data model 1 model 2 model 3
average sum model 3
each new model fixes what the team still gets wrong
model 4
Intuition
Bagging asks many experts at once and averages them; boosting asks a chain of tutors, each fixing
what is still wrong. The experts can be any kind of model (trees are the favorite).
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 7 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Our five examples, one role each
ten points eight apartments two moons breast cancer a noisy wave
(by hand) (by hand) (pictures) (real data) (curves)
rent
area
ten points, eight apartments session 08’s examples, to compute by hand
two moons to see decision boundaries
breast cancer a real dataset: 569 tumors, 30 measurements each
noisy wave to draw curves (the same wave as the boosting animation)
A small badge in the top-right corner of each frame names the data it uses.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 8 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A tiny experiment: three voters, ten examples
examples (each voter is wrong on 3 of them)
1 2 3 4 5 6 7 8 9 10
voter A 7/10
voter B 7/10
voter C 7/10
majority 10/10
Every voter is right on 7 of the 10 examples (70%).
On each example, the team follows the majority (2 or 3 votes).
The team is right on all ten: each mistake was outvoted by the other two.
Watch out
This is the best case: the mistakes never land on the same example. Does the team always win like
this?
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 9 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
. . . but voting can fail
mistakes that overlap two at a time
1 2 3 4 5 6 7 8 9 10
voter A 7/10
voter B 7/10
voter C 7/10
majority 6/10
Mistakes that overlap two at a time: the team is worse than every one of its voters (6/10).
Key idea
Same voters, same 70% each, three different teams. What decides is where the mistakes fall: a team
helps only when its members’ mistakes rarely land on the same examples.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 10 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Counting it: voters who make independent mistakes
Your turn (in pairs): three voters, each 1.00
right with probability 0.7, each on their own. 0.983
each voter right 70%How often is the majority right? right) 0.75 0.784 is each voter right 40%
coin flip all three right: 0.73 = 0.343 0.50 0.352
exactly two right: 3 · 0.72 · 0.3 = 0.441
0.25 0.154 majority right: 0.784 P(majority
With 25 such voters: 0.983. But with voters 0.00 0 10 20 30 40 50
right only 40% of the time: 0.352 for 3, and number of independent voters
0.154 for 25.
Key idea
A team beats its members when (1) each member is better than a coin flip, and (2) the members
make their mistakes independently. Break (1) and the crowd gets worse as it grows.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 11 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Averaging numbers: the noise cancels data: the noisy wave
one tree (spread 0.30) The experiment: fit a deep tree to a fresh
10 trees, OWN data (0.09) noisy dataset, and record its prediction at
100 one point, x = 1. Repeat 300 times.
truthcount One tree: the predictions spread
50 widely (0.30).
The average of 10 trees, each with
0 its own data: the spread shrinks to 0.5 1.0 1.5 2.0 √
prediction at x = 1 (300 repeats) 0.09, about 0.30/ 10.
Going deeper (optional)
For B independent predictions, each with variance σ2, the average has variance σ2/B: the spread √
falls like 1/ B. Averaging keeps the center and tightens the cluster.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 12 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What averaging can and cannot fix
scattered clustered off-center
The star is the average of the darts.
Scattered around the center (high
variance): the average lands near the
center. Averaging fixes variance.
Clustered off the center (high bias):
the average is just as far off.
Averaging cannot fix bias.
average: near the center average: still off
Key idea
So average models that are right on average but jumpy: deep trees. To move a cluster that is
off-center you need a different kind of team: boosting, after the break.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 13 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: write down your guess
Example: check yourself
We are about to average 100 copies of a model, each
trained on a slightly different version of the same data. ?
Which one will improve the most?
(a) a deep decision tree;
(b) a linear regression (a straight line).
Write your answer down. We check it in ten minutes.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 14 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where do 100 datasets come from? The bootstrap data: the ten points
We have one dataset. The trick: draw new ones from it. Draw n examples with replacement:
put each one back before the next draw.
original 1 2 3 4 5 6 7 8 9 10
×2 ×2 ×2
resample 1 1 2 3 4 5 6 7 8 9 10 3 left out
×2 ×2 ×2 ×2
resample 2 1 2 3 4 5 6 7 8 9 10 4 left out
×3 ×2
resample 3 1 2 3 4 5 6 7 8 9 10 3 left out
Some come out twice (×2) or three times (×3), some never (dashed).
Example: your turn Key idea
Ten numbered cards (or a 10-sided die). Draw Each resample leaves out a bit over a third:
ten times, putting the card back each time. 34.9% on average with 10 examples, 36.8% with
How many numbers never came out? many (here 3, 4 and 3 of 10). Remember the
left-out ones.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 15 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bagging = Bootstrap AGGregatING
bootstrap
deep tree 1
sample 1
original bootstrap vote one
deep tree 2
data sample 2 (or average) prediction
bootstrap
deep tree 3
sample 3
step 1: resample step 2: train step 3: combine
Each tree sees a slightly different dataset,
so each makes different mistakes. Intuition
Any model fits in the boxes, but use deep Classification: majority vote. Regression:
trees: right on average and jumpy, exactly
average.
what averaging fixes. Bagging adds the sta-
bility; the trees bring the accuracy.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 16 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bagging by hand: the ten points data: the ten points
tree 1 tree 2 tree 3 tree 4 tree 5 majority vote
first: x1 8.5 first: x2 5 first: x1 3.5 first: x1 3.5 first: x1 3 10 of 10 right
Filled: in the tree’s bootstrap sample. Hollow: left out of it.
Five bootstrap samples, five trees, and three different first splits (trees 3–5 all separate
P1–P3 from the rest, like session 08’s x1 ≤3.5).
The majority vote of the five is right on all ten points. . .
. . . but these are points most of the trees were trained on. Is that an honest score?
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 17 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
An honest score for free: out-of-bag data: the ten points
Each point was left out by some
P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 trees (orange). Let only those
tree 1 in in in in in out out out in in trees judge it.
tree 2 in out out out in in in in in out On the ten points: 7 of 10
tree 3 in out in in out in in in out in right — not 10 of 10. (A tie,
tree 4 out out in in out in in out in in one vote each way, goes to circle, as
tree 5 in in out out in in out in in out in scikit-learn.)
On the moons, with 300
verdict bagged trees: out-of-bag
tie tie tie tie accuracy 0.911, test accuracy
0.908.
Key idea
The out-of-bag score is a free validation set: no data held back, a close estimate of the test accuracy
(slightly pessimistic when there are few trees).
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 18 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bagging on the moons data: the two moons
Predict first: will 100 bagged trees score higher than 10?
1 tree: test 0.867 10 trees: test 0.875 100 trees: test 0.908
One tree: 0.867. Ten bagged trees: 0.875. A hundred: 0.908 (4 more of the 120 test
points right).
Honest note: a carefully pruned tree (session 08) also reaches 0.908 here. Bagging gets
there with no pruning search, and its predictions no longer jump around.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 19 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
When bagging does not help (your guess, checked) data: the noisy wave
a straight line: no change a deep tree: error −56%
2
0
y
2
truth one model 100 bagged truth one model 100 bagged
4
2 0 2 2 0 2
x x
Linear regression: the average of 100 straight lines is a straight line. Nothing improves
(the error even rises by 0.4%): its problem is bias, not variance.
Deep tree: the error drops by 56%. Its problem was variance.
Key idea
Bagging takes any model (here a straight line and a tree), but only unstable ones gain. Bagging a
stable model is wasted computation.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 20 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: 1,000 trees instead of 100? data: the noisy wave
Example: check yourself
A bagged model with 100 trees works well. 0.105
A friend wants 1,000 trees. Can the extra 0.100 trees make the model overfit? (MSE) 0.075
0.049No. The test error falls fast and then settles 0.050 0.041near 0.04: adding trees only makes the error
average more stable. 0.025 test
0.000
Watch out 1 10 100 300
Each tree can still be overgrown. If the en- bagged trees (log scale)
semble overfits, tune the tree size (e.g. the
minimum leaf size), not the number of trees.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 21 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Meet the real data: breast cancer data: breast cancer (real data)
0.3 569 tumors, 30 measurements each
(radius, texture, concave points, . . . ),points taken from images of cell nuclei.
0.2 The task: malignant (212) or benign
(357)?concave 0.1 The plot shows only 2 of the 30
benign (357) measurements.
malignant (212)worst 0.0
10 20 30 Intuition
worst radius With 30 features there are many possible first
questions. Do the bagged trees use them?
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 22 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The problem: bagged trees are too alike data: breast cancer (real data)
bagging: 5 random forest: 14
different first questions different first questions
worst perimeter 46 worst perimeter 20
worst concave points 21 worst radius 14 12 concave pointsmean concave points 11 worst mean concave points 10 worst area 11 mean concavity 9
worst radius 11 9 other features 35
0 20 40 0 20 40
trees (of 100) trees (of 100)
Bagging: 46 of the 100 trees ask the same first question (“worst perimeter”); only 5
different first questions appear.
The same few strong features (tumor size, concave points) win in every bootstrap sample,
so the trees look alike, and alike trees make the same mistakes, like voters who copy each
other.
Preview: a random forest (coming up) spreads the first question over 14 features; the
most common one starts only 20 of the 100 trees.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 23 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 24

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why alike trees limit the team data: the noisy wave
one tree (spread 0.30) Back to the spread of the predictions at
25 bagged, ONE dataset (0.20) x = 1:
100 10 trees, OWN data (0.09)
truth 10 trees with their own data:count spread 0.09.
50
Bagging, 25 trees from one
dataset: the spread only falls to
0 0.20. The trees share their data, so
0.5 1.0 1.5 2.0
prediction at x = 1 (300 repeats) they share part of their errors.
Going deeper (optional)
If every pair of the B members has correlation ρ (each with variance σ2), their average has variance
ρ σ2 + 1−ρB σ2. More trees shrink only the second term; the floor ρ σ2 goes down only if the trees
become less alike.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 24 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 25

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The random-forest trick: bench the star player
A random forest is bagging plus one
change:
the 30 measurements
split 1 at every split, the tree may look
only at a random handful of the
features, and picks the bestsplit 2
question among those (the trick
acts on the splits of a tree, so a
split 3
forest is always made of trees);
√
at every split: a new random handful of 5 (≈ 30) sometimes the star feature is “on
the bench”, so other trees learn to
use other clues.
Intuition
Like a coach who benches the star player in some practice games: the others learn to score too, and
the team becomes harder to beat. Default handful: √p features for classification (for regression
scikit-learn uses all p; set max_features yourself, e.g. p/3).
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 25 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 26

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Does it help? Ten splits of the real data data: breast cancer (real data)
bagging forest
two trees make the same mistake 27.9% 25.1% lower in all 10 splits
bagging accuracy of one member 92.2% 92.0% about the same
accuracy of the team 95.6% 96.2% better in 7, tied in 2
forest 100 trees each, averaged over 10 train/test splits (5-fold cross-validation, run twice). “Same
mistake”: of the patients that at least one of two trees gets wrong, the share that both get
wrong.
Key idea
The trees are not better, they are more different: each is about as good, but they share fewer
mistakes, so the vote corrects more of them. A modest gain here, and the forest is the usual default.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 26 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 27

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bonus: which measurements matter? data: breast cancer (real data)
optional
A forest also reports how much each
worst radius feature helped its trees: the total
worst concave points improvement from every question that
worst perimeter used it.
worst area
A quick, free ranking of the
mean concave points features.
mean concavity
mean perimeter Watch out
mean area Features that carry the same informa-
tion (radius, perimeter, area) share the
0.00 0.05 0.10 credit. And “useful for prediction” is
importance not “causes the disease”.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 27 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 28

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself
Example: check yourself
Why does a random forest hide features at random?
(a) to make each tree more accurate; ?
(b) to make the trees more different from each
other.
(b). Each tree is about as good (92.2% vs 92.0%), but the
trees share fewer mistakes, so the team improves.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 28 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 29

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Break — the first half in one picture
resample + random features
data resample + random features vote
resample + random features
Averaging cancels variance when the members make different mistakes.
Bagging makes them different with bootstrap samples; random forests add random
features at each split.
One tree on the moons: 0.867; bagging: 0.908. On the real data, a forest: 96.2% (its trees
alone: 92.0%).
Key idea
After the break: what if the models are stable but too simple, off-center on average (high bias)?
Averaging cannot help. We need models that correct each other.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 29 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 30

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A different idea: learn from your mistakes
How do you study for an exam? You take a
practice test, then you study the questions
you got wrong, take another test, and
repeat. model 1
Boosting builds its models one after the still wrong
other. model 2 team
still wrong
Each new model concentrates on what model 3
the team still gets wrong.
The members are deliberately weak
(tiny trees); the team is strong.
Key idea
Bagging averages experts trained side by side and reduces variance. Boosting adds up corrections
and reduces bias: it makes a strong model out of weak ones.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 30 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 31

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The weak learner: a stump data: the ten points
x1 3.5: 8 of 10 right
A stump asks one question. Take session 08’s P7
root: x1 ≤3.5.
P3
Right on 8 of the 10 points; P4 and P5 P6
(circled) are wrong.
P10
P9 No single question does better (one other,
x2 ≤5.5, ties). A stump is too simple to get P4
all ten: high bias.
P1
P5 Intuition
Averaging copies of this stump would not help
P8 much: they would all ask nearly the same ques-
P2 tion. Boosting does something else.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 31 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 32

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
AdaBoost, round 1: make the mistakes heavier data: the ten points
round 1: x1 3.5 new weights: the misses
(2 misses) carry 50% of the weight
Start: every point weighs 1/10. The stump x1 ≤3.5 (tied for best) misses P4 and P5.
Update: rescale so the misses carry half of all the weight and the hits the other half: the
two misses weigh 1/4 each, the eight others 1/16 each.
On these new weights, the stump just used is no better than a coin flip: the next stump
must look somewhere else.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 32 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 33

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
AdaBoost, round 2 data: the ten points
round 2: x2 5.5 new weights: the misses
(2 misses) carry 50% of the weight
On the new weights the best question is x2 ≤5.5 — session 08’s second question. It gets
the heavy points P4 and P5 right.
Predict: each stump gets a vote, and stump 2’s is bigger (fewer weighted mistakes). With
two stumps voting, is the training error lower than with one?
No: still 2 of 10 wrong (now P3 and P8). Where the stumps disagree the bigger vote wins,
so for now the team just copies stump 2. Round 3 adds a voter that settles it.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 33 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 34

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
AdaBoost, round 3: the weighted vote data: the ten points
round 3: x2 2.5 weighted vote (0.69, 0.97, 0.76)
(5 misses) all 10 right
Round 3 asks x2 ≤2.5. It misses 5 points — but they are light.
The three stumps vote with weights 0.69, 0.97, 0.76: stumps that were more accurate (on
their weights) vote louder. Now all ten points are right (training error 0.2 →0.2 →0).
The three questions are exactly session 08’s tree, found by a team of stumps.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 34 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 35

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
AdaBoost in four steps
1 Give every example the same weight.
2 Fit a weak model (usually a stump) to the Going deeper (optional)
weighted data. With weighted error ε, the vote is α =
1 a small error, 2 ln 1−εε : a large vote for 3 Give the stump a vote: the fewer (weighted) 1
zero for a coin flip (ε = 2). mistakes, the louder.
4 Make its mistakes heavier, its successes
lighter. Back to 2.
The final model is the weighted vote of all the
stumps.
Watch out
A mislabeled example keeps getting missed, so its weight grows round after round until later stumps
bend to fit it. Boosting is sensitive to noisy labels; bagging and forests are more robust.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 35 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 36

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
From AdaBoost to gradient boosting
Same idea, two ways to point at “what is
still wrong”:
AdaBoost: make the missed
examples heavier. shot 1
Gradient boosting: make the 2
leftover error the next model’s 3
target.
Intuition left after shot 1
Golf: the first shot aims at the hole. Every
next shot starts where the ball is now and
covers what is left, with a shorter swing.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 36 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 37

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Gradient boosting by hand: stage 0 data: the eight apartments
stage 0: squared errors add up to 138
15 Eight apartments: predict the rent from +5.5
+4.5 the area. +3.5
+2.5
10 Stage 0: predict the average, 7.5, for
predict 7.5 for everyone everyone.
rent
5 Residual (the leftover error) = true
2.5 value −prediction: what is still
3.5
4.5 wrong. Here from −5.5 to +5.5.
5.5
0 The squared residuals add up to 138.
1 2 3 4 5 6 7 8 That is the error to beat.
area
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 37 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 38

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Stage 1: fit a stump to the residuals data: the eight apartments
1 The residuals become the new
the new target: what is still wrong target.
+5.5
+4.5 5 +3.5
+2.5
0residual
-2.5
-3.5 5 -4.5
-5.5
1 2 3 4 5 6 7 8
area
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 38 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 39

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Stage 1: fit a stump to the residuals data: the eight apartments
1 The residuals become the new
a stump on the residuals: area 4.5? target.
5 +4 2 Fit a stump to them: area ≤4.5? It answers −4 or +4.
0residual
5 -4
1 2 3 4 5 6 7 8
area
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 38 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 40

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Stage 1: fit a stump to the residuals data: the eight apartments
1 The residuals become the new
7.5 + half of the stump: error 138 42 target.
2 Fit a stump to them: area ≤4.5? It
answers −4 or +4.
10 3 Add half of it (a learning rate of
rent 0.5): 7.5 −2 = 5.5 on the left, 7.5 + 2 = 9.5 on the right. The error
5 drops from 138 to 42.
before: 7.5
after: 5.5 | 9.5 Watch out
0 Why only half of the step? Small steps,
1 2 3 4 5 6 7 8 many of them, generalize better. We
area come back to this on the learning-rate
slide.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 38 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 41

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Stages 2 and 3: repeat data: the eight apartments
150 138 after stage 1 12.5
after stage 2 dashed: the best 10.0 after stage 3 error 100 single split (10)
rent 7.5
5.0 50 42 training
2.5 18 9.9
0.0 0
1 2 3 4 5 6 7 8 stage 0 1 2 3
area
Each stage fits a stump to the new residuals and adds half of it: 138 →42 →18 →9.9.
After three tiny stumps, the sum fits the training data better than the best single split of
session 08 (error 10). A sum of stumps can draw a staircase with more steps than any one
stump.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 39 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 42

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Watch it learn data: the noisy wave
The same recipe on the noisy wave,
from 1 stage to 120: each stage fits
a small tree to what is still wrong
and adds 0.3 of it.
Early stages capture the shape.
Late stages start bending
toward individual noisy points.
Going deeper (optional)
For classification the residual is y −ˆp
(label minus predicted probability). In
general it is the negative gradient of
the loss — hence the name.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 40 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 43

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
More rounds: bagging settles, boosting can overfit data: the noisy wave
bagging: more trees, then it settles boosting: too many rounds overfit
bagging (deep trees) learning rate 1.0
learning rate 0.1 0.10 (MSE)
error 0.05
test best: 6 rounds best: 40 rounds 0.00
1 10 100 300 1 10 100 300
trees (log scale) boosting rounds (log scale)
Bagging (left): more trees, and the test error settles.
Boosting (right) never stops fitting the training data. With learning rate 1.0 (full steps)
the test error is lowest after 6 rounds; then the model fits the noise, and after 300 rounds
(0.093) it is as bad as one overgrown tree.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 41 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 44

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The learning rate, and when to stop data: the noisy wave
learning rate 1.0
learning rate 0.1 With learning rate 0.1, each round adds
0.10 only a tenth of the correction.(MSE)
error 0.05 It40),needsbut morethat best,rounds0.025,(its bestis theis lowestafter
test error of all.
test best: 40 rounds
Like adding salt to a soup a pinch at a
0.00
1 10 100 300 time: many small corrections usually
boosting rounds (log scale) beat a few big ones.
Key idea
The recipe: a small learning rate, many rounds, and early stopping — keep part of the training data
aside, and stop adding rounds when the error on it stops improving.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 42 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 45

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself
It is overfitting. Try:
a smaller learning rate with early
Example: check yourself stopping;
Your gradient-boosting model is right on 99% of fewer rounds or shallower trees;
the training examples and 80% of the test exam-
ples. What would you change? a look at the labels: are some of
them wrong? (Boosting chases
them.)
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 43 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 46

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Any model inside: does it help? data: breast cancer (real data)
Bagging and AdaBoost are wrappers: you choose the model inside (here a tree, optional
naive Bayes, a k-NN = k nearest neighbors, or a logistic regression). Predict first:
which ones improve?
bagging (100 copies) AdaBoost (100 rounds)
deep tree +2.9 pts stops after 1 round
naive Bayes no change +2.4 pts
k-NN no change refused: no example weights
logistic regression no change 0.8 pts
0.92 0.94 0.96 0.98 1.00 0.92 0.94 0.96 0.98 1.00
test accuracy (10 splits) test accuracy (10 splits)
Bagging helps only the jumpy deep tree (better in all 10 splits).
AdaBoost lifts the too-simple naive Bayes (8 of 10 splits), makes the strong logistic
regression slightly worse, refuses k-NN, and stops at once with a deep tree.
Any model can go in, but bagging fixes jumpy models and boosting fixes too simple
ones.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 44 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 47

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Teams of different models: voting and stacking data: breast cancer (real data)
The members do not have to be the same
deep tree 0.927 kind of model. (k-NN: predict the majority
naive Bayes 0.932 label of the k closest training examples.)
k-NN 0.966 Voting: each model votes (hard), or the
SVM 0.974 team averages their probabilities (soft).
logistic regression 0.975
hard voting 0.975 Stacking: a small model on top learns
soft voting 0.972 how much to trust each member.
stacking 0.976 Here every team lands within half a
0.90 0.95 1.00 point of the best member (0.975), and
test accuracy, mean ± sd you did not need to know in advance
which member that was.
Key idea
Voting and stacking accept any mix of models. Like every team, they gain most when the members
make different mistakes.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 45 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 48

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which methods work with any model?
what can go inside in scikit-learn
bagging any model BaggingClassifier
voting any models, of different kinds VotingClassifier
stacking any models, plus one on top StackingClassifier
AdaBoost only models that take example AdaBoostClassifier
weights (in scikit-learn)
random forest trees only RandomForestClassifier
gradient boosting small trees (in scikit-learn) HistGradientBoostingClassifier
Watch out Key idea
scikit-learn’s k-NN cannot take example Every one of them has the same use: fit,
weights, so its AdaBoost refuses it with an then predict. Regression versions exist too
error. Bagging, voting and stacking accept it. (...Regressor).
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 46 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 49

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bagging vs boosting at a glance
bagging / random forest boosting
members are built in parallel, independently in sequence, each fixing what is left
members are deep trees (low bias, jumpy) small trees (stable, too simple)
combined by vote / average weighted sum
mainly fixes variance bias
more trees never hurts; it settles can overfit: stop early
noisy labels robust sensitive
tuning little (works with defaults) more (learning rate, rounds, depth)
Intuition
A committee of experts trained side by side vs a chain of tutors, each fixing what the team still gets
wrong.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 47 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 50

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Head to head on the real data data: breast cancer (real data)
optional
Predict first: rank one tree, bagging, a
one tree 0.927 random forest and gradient boosting.
bagging 0.955 Every ensemble beats the single
random forest 0.965 tree, by about 3 to 4 points (0.927
gradient boosting 0.963 against 0.955–0.969).
HistGradientBoosting 0.969 Between the ensembles the gaps
are at most 1.4 points, about the
0.90 0.95 1.00 split-to-split spread: no clear
accuracy, mean sd ± winner here.
The same 10 splits as the forest table; bagging and the forest use 200 trees. HistGradientBoosting is scikit-learn’s
fast version of gradient boosting.
Watch out
“Method X always wins” is not a thing. Try a forest and a booster, and compare them with cross-
validation on your own data.
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 48 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 51

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which one should I use?
For tabular data (a table of rows and columns), classification or regression:
must the decisions be yes a single small tree
explained as a few rules? (accept less accuracy)
no
start with a random forest
(500 trees, defaults): a
strong baseline in minutes
gradient boosting
need the last bit of accuracy, yes many wrong no HistGradientBoosting, XGBoost, LightGBM
and have time to tune? labels? small learning rate
yes + early stopping
no: the forest is your model
keep the forest:
boosting chases noise
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 49 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 52

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
In practice: the knobs that matter, and the code
Random forest
n_estimators: more trees, more stable, never from sklearn.ensemble import (
worse (a few hundred). RandomForestClassifier,
HistGradientBoostingClassifier)
max_features: features offered at each split, the
decorrelation knob.
rf = RandomForestClassifier(
min_samples_leaf: the smallest leaf, i.e. the size n_estimators=500, n_jobs=-1)
of each tree. rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))Gradient boosting
learning_rate: the size of each correction gb = HistGradientBoostingClassifier(
(small: 0.05–0.1). learning_rate=0.1, max_iter=1000,
max_iter (or n_estimators): the rounds; let early_stopping=True)
early stopping choose. gb.fit(X_train, y_train)
print(gb.score(X_test, y_test))
max_depth: keep each tree small (2–5).
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 50 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 53

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Cheat sheet
members differ because built / combined fixes main knobs main risk
bagging each sees a bootstrap in parallel / vote variance tree size (more useless on
(any model) sample or average trees only add stable models
stability)
random forest + a random handful of in parallel / vote variance max_features, hard to read
(trees only) features at each split or average number of trees (hundreds of
trees)
AdaBoost each focuses on what in sequence / bias rounds, tree depth noisy labels get
(models that the team still misses weighted vote heavy
take weights) (weights)
gradient each fits the residuals in sequence / sum bias learning rate, overfits if run
boosting (leftover errors) of corrections rounds, depth too long
(trees, in
scikit-learn)
Key idea
Independent errors cancel; shared errors do not (bagging, forests). Weak learners add up when
each one fixes what is left (boosting).
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 51 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


#### Página física 54

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Exit ticket
1. a bootstrap sample of the data; what the 1 Bagging trains every model on
previous models still get wrong. ; boosting trains each model
2. Their problem is bias, not variance: the on .
average of straight lines is a straight line.
2 Why does averaging 100 linear regressions
3. The forest only gets more stable; the
not help?
booster may overfit, so stop it early.
3 You add 1,000 trees to a random forest, 4. Bagging and stacking (and voting). A
and 1,000 rounds to a gradient-boosting forest is made of trees, and scikit-learn’s
model. What happens to each? AdaBoost needs example weights, which its
4 Which of these can be built around a k-NN k-NN cannot use.
model: bagging, a random forest,
AdaBoost, stacking?
Jan Hurtado (UTEC) Ensembles, the intuitive version October 11, 2026 52 / 54
```

**Visual:** ramas paralelas hacia «vote/average» representan bagging; flechas en cadena hacia «weighted sum» representan boosting. Marcas ✓/✗ indican predicción correcta/incorrecta; los colores diferencian aciertos y errores.


### 3. ML_Course_Clustering_Final.pdf — 74 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 10 —
Clustering
The intuitive version: finding groups when
nobody gives the answers
Jan Hurtado
UTEC
October 15, 2026
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Let’s start with a sort data: the eight objects
Example: your turn (1 minute)
Split these eight objects into two groups. Nobody
tells you what the groups should be. Compare with
your neighbor.
By color, by shape, by size: three different answers,
and all three are right. And they have nothing in
common: each color group holds two circles and two
squares, two small and two large.
Key idea
The groups depend on what you look at, and there is
no answer key. That is clustering: finding groups in
data that comes without answers.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 2 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What changes today: no labels data: 2-D shapes
what we are given: what we want back:
points, and no labels groups (and a center for each)
Sessions 02–09 were supervised: every example came with its answer y, and we learned to
predict it.
Today is unsupervised: only the features x. The task is to find structure: groups of
similar examples (clusters).
The output: a group number for every point, and usually a center for every group. To test
a method in class, we use made-up points whose groups we secretly know.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 3 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where clustering is used
Example: customer segments Example: fewer colors in a photo
Shoppers with similar habits form a Similar colors are grouped and re-
placed by one: a smaller image (later today).segment; each segment gets its own offers.
Example: news and documents Example: spotting the unusual
Articles about the same topic land A transaction that belongs to no
in the same group, with nobody tagging them. group is worth a second look.
Intuition
In each case nobody has the answers in advance. Clustering proposes groups; a person decides
whether they are useful.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 4 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What makes a good group?
Close inside: the points of one group are near
each other.
Far apart: different groups are far from each
other.
“Close” needs a distance. For numbers,
usually the straight-line distance: differences in
each feature, squared, added up, then the
square root (Pythagoras).
Watch out
inside a group: short distances (gray) A distance adds up differences of different features
between groups: long distances (arrows) — and it depends on their units. Keep that in mind:
it comes back twice today.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 5 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Today: four ideas of a group
a center a cloud a crowd a branch
the points around a it can stretch and tilt; a crowded region, of of a tree of merges;
center a point can belong a any shape; the lonely you cut the tree where
k-means little to two clouds points are noise you like
Gaussian mixture DBSCAN hierarchical clustering
Intuition
Four answers to one question: what is a group? Each answer finds some shapes and misses others.
By the end of today you will know which one to use when.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 6 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Our examples, one role each
the eight objects the eight apartments three squares the ten points five letters
the opening sort k-means by hand bad starts DBSCAN by hand scoring by hand
B
2 13 D
A C
E
2-D shapes wine (real data) iris (real data)
what methods assume a photo 178 wines, 13 numbers 150 flowers, 4 lengths
k-means on colors
Top row: small examples to work by hand. Bottom row: made-up 2-D points to see what each method does, and
real data. A small badge in the top-right corner of each frame names the data it uses.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 7 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The k-means idea
Choose K, the number of groups. Start: K centers
placed at random (for instance, at K random
points). Then repeat two moves:
1. join: every point
joins its nearest center
2. move: every center moves
to the average of its points
Stop when no point changes group. the black trails: where the three centers traveled
Intuition
K pizzerias open in a new town. Every customer goes to the nearest one; then every pizzeria moves
to the middle of its customers. Repeat until nobody switches.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 8 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
By hand: the eight apartments, pass 1 data: the eight apartments
Session 08 predicted these rents from the area; today only the rents remain: 2, 3, 4, 5, 10, 11,
12, 13. No areas, no labels. K = 2, starting centers at 0 and 9 (we pick them to keep the
arithmetic clean).
start: two centers, at 0 and 9
0 9
2 3 4 5 10 11 12 13
Your turn (2 minutes): which center is nearest to each rent?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 9 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
By hand: the eight apartments, pass 1 data: the eight apartments
Session 08 predicted these rents from the area; today only the rents remain: 2, 3, 4, 5, 10, 11,
12, 13. No areas, no labels. K = 2, starting centers at 0 and 9 (we pick them to keep the
arithmetic clean).
pass 1, step 1: each rent joins its nearest center
0 9
2 3 4 5 10 11 12 13
border 4.5
Rents 2, 3, 4 are nearer to 0; rents from 5 up are nearer to 9. The border is halfway between
the centers, at 4.5.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 9 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
By hand: the eight apartments, pass 1 data: the eight apartments
Session 08 predicted these rents from the area; today only the rents remain: 2, 3, 4, 5, 10, 11,
12, 13. No areas, no labels. K = 2, starting centers at 0 and 9 (we pick them to keep the
arithmetic clean).
pass 1, step 2: each center moves to the average of its rents
(2+3+4)/3 = 3 (5+10+11+12+13)/5 = 10.2
3 10.2
2 3 4 5 10 11 12 13
border 4.5
Each center moves to the average of its rents: 3 and 10.2.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 9 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Passes 2 and 3: repeat until nothing changes data: the eight apartments
pass 1, step 2: each center moves to the average of its rents
(2+3+4)/3 = 3 (5+10+11+12+13)/5 = 10.2
3 10.2
2 3 4 5 10 11 12 13
border 4.5
Predict first: with the centers at 3 and 10.2, which rent changes sides?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 10 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Passes 2 and 3: repeat until nothing changes data: the eight apartments
pass 2, step 1: the border moves to 6.6, and rent 5 changes sides
3 10.2
2 3 4 5 10 11 12 13
border 6.6
Rent 5: it is 2 from 3, but 5.2 from 10.2. The border moves to 6.6.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 10 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Passes 2 and 3: repeat until nothing changes data: the eight apartments
pass 2, step 2: the centers move to 3.5 and 11.5
(2+3+4+5)/4 = 3.5 (10+11+12+13)/4 = 11.5
3.5 11.5
2 3 4 5 10 11 12 13
border 6.6
The new averages: 3.5 and 11.5.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 10 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Passes 2 and 3: repeat until nothing changes data: the eight apartments
pass 3: border 7.5, nobody changes sides stop
3.5 11.5
2 3 4 5 10 11 12 13
border 7.5
Pass 3: the border is 7.5, nobody changes sides, so we stop.
Key idea
Without any labels, k-means split the rents exactly where session 08’s tree split them with labels:
two groups centered at 3.5 and 11.5.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 10 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What k-means is minimizing data: the eight apartments
The cost: every point pays the squared distance
to its own center; add it all up. It is called the one center for all eight rents
WCSS (within-cluster sum of squares).
250 the average, 7.5 Why the average? One center for all eight
cost 200 156 156 rents is cheapest at their average, 7.5: session 08’s best constant (cost 138).
150
140 140
138 So move can only lower the cost, and so can
100
4 5 6 7 8 9 10 11 join (every point picks its nearest center).
where the one center sits
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 11 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What k-means is minimizing data: the eight apartments
The cost: every point pays the squared distance
to its own center; add it all up. It is called the
the cost after every step
WCSS (within-cluster sum of squares).
75
75
Why the average? One center for all eight
50 40.8 rents is cheapest at their average, 7.5:(WCSS)
session 08’s best constant (cost 138).
cost 25 17.76 10 10 So move can only lower the cost, and so can
0
pass 1 pass 1 pass 2 pass 2 pass 3 join (every point picks its nearest center).
join move join move join
The cost never goes up:
75 →40.8 →17.76 →10 →10.
Going deeper (optional)
WCSS = Pk Pi∈group k∥xi −ck∥2. For one group, Pi(xi −c)2 is smallest where its slope
−2 Pi(xi −c) is zero: at c = the average.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 11 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Watch it learn data: 2-D shapes
Three centers start badly placed,
below the data (this start recovers;
not every start does). Pale colors:
each center’s territory.
Each pass: join (colors change),
then move (centers jump; the
trails show their path).
The borders: straight lines
halfway between two centers.
The cost never goes up. After 7
passes nothing changes: stop. It
always stops: there are only
finitely many groupings.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 12 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Real data: a photo with 96,615 colors data: a photo
original: 96,615 different colors k-means with K = 2: 2 colors
Each of the 273,280 pixels is three numbers: red, green, blue. k-means groups those colors into
K groups and repaints every pixel with its group’s center. Predict first: how many colors before
you stop noticing?
(photo: danielbuechele (Flickr), CC BY 2.0, via scikit-learn; colors reduced with k-means)
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 13 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Real data: a photo with 96,615 colors data: a photo
original: 96,615 different colors k-means with K = 2: 2 colors
K = 2: just light and dark, a two-tone poster.
(photo: danielbuechele (Flickr), CC BY 2.0, via scikit-learn; colors reduced with k-means)
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 13 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Real data: a photo with 96,615 colors data: a photo
original: 96,615 different colors k-means with K = 4: 4 colors
K = 4: no red left at all.
(photo: danielbuechele (Flickr), CC BY 2.0, via scikit-learn; colors reduced with k-means)
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 13 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Real data: a photo with 96,615 colors data: a photo
original: 96,615 different colors k-means with K = 16: 16 colors
K = 16: the red-orange roofs are back. Each pixel now stores a 4-bit color number instead of 24
bits: 6 times smaller than the raw bitmap.
(photo: danielbuechele (Flickr), CC BY 2.0, via scikit-learn; colors reduced with k-means)
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 13 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Real data: a photo with 96,615 colors data: a photo
original: 96,615 different colors k-means with K = 64: 64 colors
K = 64: close to the original (4 times smaller). Here K is a trade-off between quality and size
— not a “true” number of groups.
(photo: danielbuechele (Flickr), CC BY 2.0, via scikit-learn; colors reduced with k-means)
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 13 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 24

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: two runs, two answers
Example: check yourself
You run k-means twice on the same data, with the same ?
K, and get two different groupings. Is it a bug?
No. The starting centers are random, and k-means stops at
the first arrangement where nothing changes — which one
depends on the start. Next section.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 14 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 25

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bad starts: a local minimum data: three squares
start A WCSS 24 start B WCSS 216
(the best possible) (stuck: 9 times worse)
Three squares of four points, K = 3, two starts (the rings). Start A finds the squares: cost
24, the best possible.
Start B splits by rows: cost 216, nine times worse — and stuck: every point is with its
nearest center, every center at its group’s average.
Intuition
A hiker in fog stops in the first valley: every step from here goes up, yet a valley nine times deeper
exists elsewhere.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 15 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 26

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The cure: restart, and start cleverly data: three squares
Try every start (3 of the 12 points as
centers): 188 of 220 find 24; 32 get stuck,
about 1 in 7.
all 220 starts: 188 find 24 Restart: 10 random starts, keep the lowest
200 188 cost. All 10 stuck: about 4 in a billion.
100 new start at random, but far points arestarts Start cleverly: k-means++ picks each
8 10 6 5 2 1 likelier. After a first center at the
0 bottom-left point, the second lands in the 24 216 220 221.3 270 271.3 274
where it stops (WCSS) same square 1.4% of the time (a blind
pick: 27%).
Watch out
scikit-learn’s KMeans starts only once by default
(n_init="auto"). Ask for several: n_init=10.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 16 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 27

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Did it find the right groups? data: five letters
the true groups what a method found
B B
D D
A C A C E E
group 1 group 2 group 2 group 1
Only on data where we secretly know the true groups (in real use there is no key).
A method found exactly the right groups — but called them the other way round. Letter
by letter, 0 of 5 match! Group names are arbitrary.
So we compare pairs of points: together in both groupings, or apart in both?
The ARI (adjusted Rand index) turns that into one number: 1 = the same grouping (here:
1), 0 = what a random grouping gets.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 17 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 28

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Your turn: score a grouping by counting pairs data: five letters
the true groups what a method found
B B
D D
A C A C E E
Your turn (3 minutes): 5 letters give 10 pairs (AB, AC, . . . , DE). For each pair: together
in both groupings, or apart in both?
6 of 10 agree: AB and DE (together in both); AD, AE, BD, BE (apart in both).
A random grouping with the same sizes (3 and 2) agrees on 52% of the pairs, on average.
The ARI rescales so that this counts as 0: here, ARI = 0.17.
Going deeper (optional)
agreement −chance 0.60 −0.52
ARI = = = 0.17.
1 −chance 1 −0.52
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 18 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 29

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Your turn: score a grouping by counting pairs data: five letters
the true groups what a method found
B B
D D
A C A C E E
Your turn (3 minutes): 5 letters give 10 pairs (AB, AC, . . . , DE). For each pair: together
in both groupings, or apart in both?
6 of 10 agree: AB and DE (together in both); AD, AE, BD, BE (apart in both).
A random grouping with the same sizes (3 and 2) agrees on 52% of the pairs, on average.
The ARI rescales so that this counts as 0: here, ARI = 0.17.
Key idea
4 of 5 letters are “right”, yet ARI 0.17: the ARI is strict, and it is not a percentage. (The opening’s
color and shape sorts score −0.17 against each other: no agreement.)
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 18 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 30

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What k-means assumes data: 2-D shapes
round blobs two moons tilted clouds small beside big
(the true groups) (the true groups) (the true groups) (the true groups)
Predict first: k-means, with the right K, on these four datasets. Which ones will it get
right?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 19 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 31

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What k-means assumes data: 2-D shapes
round blobs two moons tilted clouds small beside big
k-means: ARI 1.00 k-means: ARI 0.23 k-means: ARI 0.36 k-means: ARI 0.21
Round blobs: perfect. Moons: 223 of 300 points in the right moon, yet ARI 0.23 (a coin
flip already gets half right).
Why? The border between two centers is the straight line halfway between them (the
rents’ 4.5, in 2-D): every group has straight sides — and a moon has a dent.
Key idea
Not a bad optimizer: on the moons, k-means’ answer has a lower cost than the true groups (122
against 183). Its cost prefers round groups of similar size, in comparable units — more restarts
cannot help.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 19 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 32

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Units matter, by hand data: the eight objects
in cm: split by shape
(cost 4 vs 16)
6
(cm) 4
2 diameter
0
0 2 4
corners
Four of the eight objects, measured by (corners, diameter). In cm, shapes are 4 apart and sizes 2
apart: k-means splits them by shape (cost 4 against 16).
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 20 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 33

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Units matter, by hand data: the eight objects
in cm: split by shape
(cost 4 vs 16)
6
(cm) 4
2 diameter
0
0 2 4
corners
Predict first: we write the diameters in mm. Do the groups change?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 20 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 34

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Units matter, by hand data: the eight objects
in cm: split by shape in mm: split by size
(cost 4 vs 16) (cost 16 vs 400)
6 50
40 (cm) 4 (mm)
30
2 20 diameter diameter
10 0
0 2 4 20 0 20
corners corners
Yes: the sizes are now 20 apart. k-means splits by size (cost 16 against 400). Nothing about
the objects changed — only the unit.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 20 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 35

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Units matter, by hand data: the eight objects
in cm: split by shape in mm: split by size standardized: a tie
(cost 4 vs 16) (cost 16 vs 400) (cost 4 and 4)
6 50
40 std. 1 (cm) 4 (mm)
30 0
2 20 1 diameter diameter diameter,
10 0
0 2 4 20 0 20 1 0 1
corners corners corners, std.
The cure: standardize. Subtract each feature’s average, divide by its spread. Corners 0/4
become −1/+1; sizes 2/4 cm become −1/+1, and so do 20/40 mm.
Key idea
The unit loses its vote: shape and size now tie (cost 4 = 4). Standardizing gives every feature the
same say; which one matters is your decision.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 20 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 36

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Meet the real data: wine data: wine (real data)
178 wines from three grape varieties
(botanists say “cultivars”), 13 chemical 2 of the 13 measurements
measurements each: alcohol,
12.5 variety 1 (59 wines) magnesium, color intensity, proline, . . .
variety 2 (71 wines)
10.0 variety 3 (48 wines) The variety is hidden from the algorithm;
7.5 we use it only to score the groups.intensity
color 5.0 Themeasurements.plot shows 2 of the 13
2.5
250 500 750 1000 1250 1500 1750 Intuition
proline Proline runs from 278 to 1,680; color intensity
from 1.3 to 13. Which of the two will decide
the distances?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 21 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 37

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Units matter on real data data: wine (real data)
raw numbers: ARI 0.37 standardized first: ARI 0.90
10 intensity
color 5
500 1000 1500 500 1000 1500
proline proline
The distance between two wines is almost all proline (99.8%). Raw, k-means just cuts
proline into three bands: 125 of 178 wines in the right group (ARI 0.37).
Standardized, proline 1,061 and color intensity 7.37 both become +1 (one typical step
above average). Now 172 of 178 (ARI 0.90).
Watch out
Different units (euros and years, proline and color)? Standardize. The same unit with meaningful
spreads (four lengths in cm)? Keep them as they are.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 22 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 38

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: euros or thousands of euros?
Example: check yourself
Customers are described by their yearly income (in eu-
ros) and their age (in years). You write the income in
thousands of euros instead. age
(a) Do k-means’ groups on the raw numbers change? ?
(b) And if you standardize first?
income
(a) Yes: income’s numbers shrink 1,000 times, down to the
size of the ages, so income no longer decides alone and age
gets a real say. (b) No: euros and thousands of euros give
exactly the same standardized numbers.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 23 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 39

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
How many groups? The elbow data: the eight apartments
one big drop, then small ones Predict first: why not try K = 1, 2, . . . , 8
150 138 and keep the K with the smallest cost?
The best cost for K = 1, 2, 3, 4: 138, 10,(WCSS) 100 the bend: K = 2 6, 2. More groups always lower it; with
cost 50 K = 8 every rent is alone and the cost is 0. “Smallest” always picks the largest K.
10 6 2best Look for the elbow: where one more 0
1 2 3 4 group stops paying off. Here K = 2: a
number of groups K drop of 128, then only 4 and 4.
Intuition
A T-shirt shop decides between S/M/L and XS to XXL; customers do not come in sizes. K is a
decision; the elbow and the silhouette (next) are hints.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 24 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 40

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The silhouette, by hand data: the eight apartments
pass 3: border 7.5, nobody changes sides stop
3.5 11.5
2 3 4 5 10 11 12 13
border 7.5
Predict first: which rents sit least comfortably in their group?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 25 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 41

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The silhouette, by hand data: the eight apartments
one rent, by hand all eight rents, averaged
1.0
b = (5+6+7+8)/4 = 6.5 0.79 0.8
a = (3+2+1)/3 = 2 0.61
0.6 silhouette 0.47
0.4
2 3 4 5 10 11 12 13
0.2 rent 5: s = (b - a) / b = 0.69 average
0.0
2 3 4
number of groups K
Predict first: which rents sit least comfortably in their group?
For one point: a = its average distance to the other points of its group; b = to the nearest
other group. Its silhouette: s = (b −a) divided by the larger of a and b. Near 1:
comfortable. Near 0: on the fence. Negative: probably in the wrong group.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 25 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 42

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
On the wine: both say three data: wine (real data)
optional
the elbow: a gentle bend at K = 3 the silhouette: highest at K = 3
0.30
2000
0.25 (WCSS) silhouette
1500 cost
0.20 best average 1000
0.15
1 2 3 4 5 6 7 8 2 3 4 5 6 7 8
number of groups K number of groups K
The elbow: big drops until K = 3 (655 and 381), then 103 and 70 — a gentle bend at 3.
The silhouette: highest at K = 3 (0.28; 0.26 at K = 2 and at K = 4).
And there are three grape varieties. Nice — but nothing guarantees it.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 26 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 43

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Clustering always answers data: 2-D shapes
500 random points: no groups random points: K = 4 scores 0.41
1.0
the eight apartments, K = 2: 0.79 0.8
0.6 silhouette
0.4
0.2 the wine, K = 3: 0.28 average 0.0
2 3 4 5 6 7 8
number of groups K
500 random points: there are no groups. k-means still returns the K groups you ask for.
And the silhouette scores K = 4 at 0.41 — higher than the wine’s real groups (0.28). Its
size cannot tell real groups from none.
Key idea
Neither k-means nor the silhouette ever says “there are no groups here”. Always check that the
groups mean something for what you need.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 27 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 44

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: the true number of groups?
Example: check yourself silhouette
Your manager wants “the true number of customer
types”. The silhouette says 4 (0.41). What do you an- ?
swer?
There may be no true number: random points also score
K
0.41. Choose K for the use (how many campaigns can you
run?), then check that the groups make sense.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 28 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 45

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Break — the first half in one picture
join: nearest center A random start can get stuck: restart
(n_init=10).
move: center The biggest numbers rule: standardize features in
to the average different units.
the cost never goes up, and it always stops K is your decision; the elbow and the silhouette
are hints.
k-means: round groups of similar size, in comparable units — fast, on any amount of data.
Key idea
After the break: groups that are not round blobs — stretched or unequal (clouds), curved (crowds),
nested (trees).
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 29 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 46

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Hard or soft? data: 2-D shapes
mixture: 87 of 420 points are
k-means: every point must choose less than 90% sure (mixed colors)
A point between two groups: k-means must give it exactly one color.
A mixture model gives each point a probability for each group, for instance 70% green
and 30% orange (a mixed color) — like a forecast of “70% rain” instead of just “rain”.
Here 87 of the 420 points are less than 90% sure of their group.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 30 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 47

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Each group as a cloud data: the eight apartments + 2-D shapes
2-D: an ellipse =
1-D: a bell = a center + a width center + widths + tilt
center 3.5 center 11.5
0 5 10 15
rent
A cloud (a Gaussian) = a center + a spread + a tilt + a size (its share of the points).
The story behind it: pick a cloud (bigger ones more often), then drop a point near its
center. A mixture tries to recover the clouds from the points.
Intuition
k-means’ groups are clouds forced to be round, equally wide and of the same size. A mixture lets
every cloud have its own width, stretch, tilt and size.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 31 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 48

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Soft membership, by hand data: the eight apartments
P(low) = 0.465 / (0.465 + 0.282) = 0.62
low bell: 0.465 high bell: 0.282
rent 7
0 2 4 6 8 10 12 14 16
rent
Predict first: a new apartment rents for 7. Low
P(low) at rent 6 7 8
or high — and how sure?
wide (variance 8) 0.82 0.62 0.38
k-means: low (7 is left of the border 7.5). narrow (variance 1.25) 1.00 0.96 0.04
Full stop.
Narrow bells: at 7 the low one is about 25
A mixture compares the bell heights at 7: times taller, so 25/26 = 96%.
0.465/(0.465 + 0.282) = 62% low. The wider the clouds, the softer the answer;
very narrow clouds give k-means’ hard answer.
At 7.5, halfway: always 50/50.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 32 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 49

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
EM: k-means with soft memberships data: 2-D shapes
Same data as the k-means animation; the
three clouds start in the middle.
1 E-step: every point works out how
much it belongs to each cloud
(mixed colors = shared points).
2 M-step: every cloud moves to the
weighted average of its points and
reshapes. A point that is 70% cloud
1 counts as 0.7 of a point there.
Join and move, only soft. Each round fits at
least as well as the last; the clouds settle.
Watch out
Like k-means, EM can stop at a poor
answer: restart it (n_init).
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 33 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 50

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where the shape freedom pays data: 2-D shapes
tilted clouds tilted clouds
k-means: ARI 0.36 mixture: ARI 0.81 small beside big small beside big
k-means: ARI 0.21 mixture: ARI 1.00
Tilted clouds: k-means 0.36. A mixture fits each cloud’s stretch and tilt: 0.81 (where
clouds cross, nobody can tell).
Small beside big: k-means splits the big one (72 and 148 points); the mixture finds 20
and 200 exactly: 1.00.
Key idea
The stretch fixes tilted groups; the size fixes small-beside-big. What pays is the shape freedom, not
the softness.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 34 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 51

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Real flowers: iris data: iris (real data)
k-means mixture optional
16 flowers misplaced, ARI 0.73 5 flowers misplaced, ARI 0.90
2.5
(cm) 2.0
width 1.5 1.0
0.5 petal
X = in the wrong group
0.0
2 4 6 2 4 6
petal length (cm) petal length (cm)
150 iris flowers, 4 measurements each, all lengths in cm (so we keep them as they are); the
3 species are hidden. The plot shows 2 of the 4.
Two species overlap, in tilted clouds. k-means misplaces 16 flowers (ARI 0.73); a mixture
only 5 (ARI 0.90).
Force the mixture’s clouds to be round, and it returns exactly k-means’ grouping (16
misplaced): the gain comes from the stretch.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 35 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 52

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: how sure?
Example: check yourself
(a) Make both bells wider. Is the apartment at 7
more or less surely “low”?
(b) A mixture says customer A is 51% / 49%
“bargain hunter” / “brand lover”, and customer B ?
99% / 1%. Whom do you treat differently — and
what did k-means say about each?
(a) Less sure: wider clouds, softer answers (62% instead of
96%). (b) A is on the fence: ask, or send both offers.
k-means gave each one a single group, with no “how sure”:
A looked as certain as B.
Gaussian mixture: stretched, tilted or unequal groups, or when you need “how sure”.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 36 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 53

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Clouds cannot bend data: 2-D shapes
two rings, mixture: ARI 0.00
two moons, mixture: ARI 0.46
Clouds can stretch, but they cannot bend. On the moons a mixture scores 0.46; on the
rings 0.00.
For crescents and rings we need another idea of a group: a crowd.
Intuition
From a drone over a festival you see crowds, not centers: a crowd is wherever people stand close
together, whatever its shape. The lone walkers in between belong to none.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 37 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 54

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
DBSCAN: clusters are crowded regions
Pick a radius ε (eps) and a number minPts.
Core: at least minPts points within ε, itself
included (minPts = 3: “me and two others”).
border
Core points within ε of each other join one cluster,
which grows like a chain.
ε Border: not crowded itself, but within ε of a core
point. It joins that cluster.
Noise: neither. It belongs to no cluster.
noise
core
Key idea
No K to choose: the number of clusters comes out of
the data (and out of ε).
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 38 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 55

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
By hand: the ten points data: the ten points
ε = 3.2, minPts = 3. Squared distances against 3.22 = 10.24:
radius 3.2 around P1, P4, P10 10 is in; 13 (P4 to P6) is out.
point neighbors (squared distance) count
P7 P1 P2 (10) 2
P3 P2 P1 (10) 2
P3 P6 (2), P7 (5) 3
P6 P10 P4 P5 (8), P8 (10), P9 (10) 4 P5 P8 (2), P4 (8), P9 (10) 4
P9 P6 P3 (2), P7 (5) 3 P4 P7 P3 (5), P6 (5) 3 P1
P5 P8 P5 (2), P4 (10) 3 P9 P10 (2), P4 (10), P5 (10) 4
P8 P10 P9 (2) 2
P2 Predict first: P1 and P2 are neighbors. Are they a
cluster?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 39 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 56

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
By hand: the ten points data: the ten points
ε = 3.2, minPts = 3. Squared distances against 3.22 = 10.24:
counts: core, border, noise 10 is in; 13 (P4 to P6) is out.
point neighbors (squared distance) count
P7 (3) P1 P2 (10) 2
P3 (3) P2 P1 (10) 2
P3 P6 (2), P7 (5) 3
P6 (3) P10 (2) P4 P5 (8), P8 (10), P9 (10) 4 P5 P8 (2), P4 (8), P9 (10) 4
P9 (4) P6 P3 (2), P7 (5) 3 P4 (4) P7 P3 (5), P6 (5) 3 P1 (2)
P5 (4) P8 P5 (2), P4 (10) 3 P9 P10 (2), P4 (10), P5 (10) 4
P8 (3) P10 P9 (2) 2
P2 (2) No: a pair is not a crowd (2 < 3). 7 core points; P10
is a border point (near P9); P1 and P2 are noise.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 39 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 57

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
By hand: the ten points data: the ten points
ε = 3.2, minPts = 3. Squared distances against 3.22 = 10.24:
two clusters, two noise points 10 is in; 13 (P4 to P6) is out.
point neighbors (squared distance) count
P7 P1 P2 (10) 2
P3 P2 P1 (10) 2
P3 P6 (2), P7 (5) 3
P6 P10 P4 P5 (8), P8 (10), P9 (10) 4 P5 P8 (2), P4 (8), P9 (10) 4
P9 P6 P3 (2), P7 (5) 3 P4 P7 P3 (5), P6 (5) 3 P1
P5 P8 P5 (2), P4 (10) 3 P9 P10 (2), P4 (10), P5 (10) 4
P8 P10 P9 (2) 2
P2 Two clusters: {P3, P6, P7} and {P4, P5, P8, P9,
P10}.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 39 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 58

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
By hand: the ten points data: the ten points
ε = 3.2, minPts = 3. Squared distances against 3.22 = 10.24:
two clusters, two noise points 10 is in; 13 (P4 to P6) is out.
point neighbors (squared distance) count
P7 P1 P2 (10) 2
P3 P2 P1 (10) 2
P3 P6 (2), P7 (5) 3
P6 P10 P4 P5 (8), P8 (10), P9 (10) 4 P5 P8 (2), P4 (8), P9 (10) 4
P9 P6 P3 (2), P7 (5) 3 P4 P7 P3 (5), P6 (5) 3 P1
P5 P8 P5 (2), P4 (10) 3 P9 P10 (2), P4 (10), P5 (10) 4
P8 P10 P9 (2) 2
P2 They cut across session 08’s classes (P1–P5 against
P6–P10): ARI −0.02. Close is not the same as same
class.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 39 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 59

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Any shape, and it sets the noise aside data: 2-D shapes
moons + 2 strays (circled):
two moons, k-means: ARI 0.23 DBSCAN ARI 1.00, strays = noise
two rings, k-means: ARI 0.00 two rings, DBSCAN: ARI 1.00, no noise
ε = 0.20, minPts = 5: both shapes found exactly (ARI 1.00), and the two stray points between
the moons are called noise. k-means: 0.23 and 0.00.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 40 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 60

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
“No K needed” is half true: ε decides data: 2-D shapes
eps 0.08: 21 clusters, 65 noise eps 0.20: 2 clusters eps 0.30: 1 cluster
Predict first: what happens with a tiny ε? With a huge one?
Too small (0.08): 21 little clusters and 65 points of noise. Too large (0.30): everything in
one cluster.
Any ε from 0.12 to 0.27 gives the two moons: trust an answer that survives a wide range
of ε.
A good ε makes nearly every point of a cluster a core point, yet stays below the gap
between clusters. A tight cluster beside a loose one may need two different ε — and
DBSCAN has only one.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 41 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 61

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: turn the knobs data: the ten points
two clusters, two noise points
Example: check yourself
Back to the ten points. P7
P3
(a) ε = 4: compare the squared distances with 16
instead of 10.24. P4 and P6 are 13 apart P6 P10
(squared). What happens? P9 P4 P1 (b) minPts = 2: what happens to P1 and P2?
P5
(a) P4 and P6 become neighbors (both are core): the two P8
P2clusters join into one. (b) A pair is now a crowd: P1 and
P2 form a third cluster, and nothing is noise.
DBSCAN: odd shapes and outliers, with few features and one density.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 42 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 62

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Merge the closest, then cut the tree data: the eight apartments
Start: every rent is its own group.
Repeat: merge the two closest
gaps: 1 1 1 5 1 1 1 groups. The distance between two
2 3 4 5 10 11 12 13 groups: the gap between their
closest members.
The tree (a dendrogram) records(gap) 5
cut here: every merge at its height: six gaps of 4
two groups 1, then the gap of 5.
3height
2 Cut between 1 and 5: two groups,
1 {2, 3, 4, 5} and {10, 11, 12, 13}.merge 0 (Below 1: 8 groups; above 5: one.)
2 3 4 5 10 11 12 13
Like a family tree read upward:
siblings first, then cousins, then the
family.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 43 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 63

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
How far apart are two groups? The linkage
single: complete: average: Ward: how much
the closest pair the farthest pair all the pairs the cost would grow
one center for all six
What we just did is single linkage: the closest pair. Complete: the farthest pair.
Average: all the pairs.
Ward: merge the two groups whose union raises the cost (WCSS) the least — the k-means
idea, built bottom-up.
Intuition
On the rents all four rules give the same two groups. On harder shapes they do not.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 44 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 64

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The linkage changes the answer data: 2-D shapes
single linkage average linkage Ward linkage complete linkage
ARI 1.00 ARI 0.43 ARI 0.30 ARI 0.20
Predict first: which rule can follow a crescent?
Single linkage follows the moons (ARI 1.00): it chains close neighbors. Average 0.43, Ward
0.30, complete 0.20 (k-means: 0.23).
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 45 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 65

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The linkage changes the answer data: 2-D shapes
single linkage + 2 strays (X):
cluster 1 = both moons, cluster 2 = the strays
Predict first: which rule can follow a crescent?
Single linkage follows the moons (ARI 1.00): it chains close neighbors. Average 0.43, Ward
0.30, complete 0.20 (k-means: 0.23).
But it is fragile: for two groups it cuts its biggest gap, and two strays farther from the
moons than the moons are apart make a bigger one: “both moons” against “the two
strays” (ARI 0.00). DBSCAN called them noise.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 45 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 66

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A graph of neighbors: spectral clustering data: 2-D shapes
optional
link the close neighbors cut the weakest links: ARI 1.00
Link each point to its close neighbors; then cut the graph where the links are fewest and
weakest. On the moons: ARI 1.00.
But “close” has a knob too: with scikit-learn’s default setting (γ = 1) it scores 0.29 —
barely better than k-means (0.23).
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 46 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 67

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: the tree
Example: check yourself
(a) A ninth apartment rents for 7.5. With single
linkage, at what height do the two groups merge
now? Cut the tree at 3: how many groups?
(b) Could you build a Ward tree for 100,000
customers? ?
(a) At 2.5: the new rent is 2.5 from both 5 and 10, a
stepping stone across the gap. A cut at 3 gives one group:
single linkage chains. (b) Hardly: Ward (like complete and
average linkage) keeps every pairwise distance — 100,000
points, 5 billion distances, 40 GB of memory. Use k-means
or DBSCAN (or a sample).
Hierarchical: small data (up to a few thousand points), when you want the tree or every K at
once.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 47 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 68

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which method for which situation? The evidence data: 2-D shapes + wine
k-means mixture DBSCAN Ward single
round blobs
two moons
two rings
tilted clouds
small beside big
wine (13-D)
Predict first: which method wins each row?
ARI, each method at its best: K = the true number of groups; DBSCAN at its best ε (minPts 5). Wine: standardized, and the mixture uses round
clouds.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 48 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 69

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which method for which situation? The evidence data: 2-D shapes + wine
k-means mixture DBSCAN Ward single
round blobs 1.00 1.00 1.00 1.00 1.00
two moons 0.23 0.46 1.00 0.30 1.00
two rings 0.00 0.00 1.00 0.00 1.00
tilted clouds 0.36 0.81 0.46 0.31 0.00
small beside big 0.21 1.00 0.82 0.24 −0.02
wine (13-D) 0.90 0.88 0.44 0.79 −0.01
No column wins every row. On the wine, single linkage puts 174 of 178 wines in one group, and
DBSCAN, even at its best, finds two crowds, not three, and calls 34 wines noise.
ARI, each method at its best: K = the true number of groups; DBSCAN at its best ε (minPts 5). Wine: standardized, and the mixture uses round
clouds.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 48 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 70

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Use it when, avoid it when
use it when avoid it when you choose
k-means round groups of similar size; curves, rings, outliers; very K
any amount of data unequal groups
mixture stretched, tilted or unequal curves and rings; many K, the clouds’
groups; you need “how measurements, few points shape
sure”
DBSCAN odd shapes; noise to set many features; groups of ε, minPts
aside; few features different densities
hierarchical small data; you want the big data (every pair of the linkage,
tree, or every K points is compared) where to cut
Watch out Key idea
All four measure distances: features in Every one of them has a knob that decides how
different units must be standardized first. many groups you get, so their groups never
prove that real groups exist.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 49 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 71

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which one should I use?
first: features in different (the same unit with meaningful spreads,
units? standardize them like four lengths in cm: keep them)
few features (you can plot yes
them), and curved shapes DBSCAN
or outliers to set aside?
no
stretched or unequal groups, yes small data, and
Gaussian mixture
or you need “how sure”? you want the tree
no or every K?
hierarchical (Ward)
k-means
(n_init=10)
Then look at the groups: do they make sense for what you need?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 50 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 72

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
In practice: the knobs that matter, and the code
StandardScaler: first, if the units differ.
from sklearn.preprocessing import StandardScaler
KMeans: n_clusters (K); n_init, the
from sklearn.cluster import KMeans number of starts (the default runs one).
GaussianMixture: n_components (K), Xs = StandardScaler().fit_transform(X)
n_init. With many measurements and few km = KMeans(n_clusters=3, n_init=10).fit(Xs)
points, full clouds can collapse onto a print(km.labels_) # a group for every row
handful of points: try print(km.inertia_) # the cost (WCSS)
covariance_type="diag" or "spherical".
DBSCAN: eps and min_samples; label −1 # the others: labels = model.fit_predict(Xs)
means noise. # from sklearn.mixture import GaussianMixture
# GaussianMixture(n_components=3, n_init=10)
AgglomerativeClustering: n_clusters # DBSCAN(eps=..., min_samples=5) # tune eps
and linkage. # AgglomerativeClustering(n_clusters=3,
# linkage="ward")
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 51 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 73

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Cheat sheet
the idea knobs watch out
k-means join the nearest center, move K, n_init bad starts; units; it always returns
(round, similar it to the average, repeat K groups
groups)
mixture clouds with a center, spread, K, n_init bad starts; curved shapes
(stretched or tilt and size; soft memberships
unequal groups) (EM)
DBSCAN clusters are crowded regions; ε, minPts many features; different densities
(any shape, with the rest is noise
noise)
hierarchical merge the closest groups, linkage, the cut big data; single linkage chains
(small data, the record a tree, cut it
whole tree)
Key idea
Close inside, far apart — measured with a distance, so mind the units. Every method proposes
groups; deciding whether they are useful is your job.
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 52 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


#### Página física 74

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Exit ticket
1. Join the nearest center; move each center 1 k-means repeats two moves:
to the average of its points. and .
2. The random start: each run stops at its
2 Two runs of k-means give two different
first stable arrangement. Run several starts
groupings. Why, and what do you do?
(n_init=10) and keep the lowest cost.
3 Income in euros, age in years: what do you 3. Standardize: distances add up the features,
do before clustering, and why? so income’s big numbers would decide alone.
4 Which method for: (a) crescents with a 4. (a) DBSCAN; (b) a Gaussian mixture; (c)
few outliers; (b) “how sure is each point?”; hierarchical clustering (the tree).
(c) groups at every level of detail?
Jan Hurtado (UTEC) Clustering, the intuitive version October 15, 2026 53 / 55
```

**Visual:** puntos de colores indican grupos asignados; cruces grandes señalan centroides; trazos/flechas indican desplazamiento de centroides entre iteraciones. Puntos aislados pueden indicar ruido en DBSCAN, solo cuando lo diga la figura.


### 3. ML_Course_regularization_and_model_selection.pdf — 36 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 03 — Regularization
& Model Selection
Controlling complexity, the bias–variance
trade-off, and choosing models
Jan Hurtado
UTEC
August 23, 2026
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Roadmap
1 Why we need to control complexity
2 The bias–variance trade-off
3 Regularization: shrinking the weights
4 Choosing the model: cross-validation
5 Putting it together: a complexity sweep
6 Real-world example
7 Wrap-up
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 2 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Key terms (for newcomers)
Hyperparameter: a setting you choose, not Shrinkage: pulling weights toward zero.
learn (e.g. λ). Sparsity: many weights exactly zero (lasso)
Bias: error from a too-simple model. — feature selection.
Variance: error from a model too sensitive to Cross-validation (fold): rotating held-out
the sample. splits to estimate error honestly.
Underfit / overfit: too simple (high bias) / Data leakage: evaluation data influencing
too flexible (high variance). training — over-optimistic scores.
Regularization: a cost on large weights; λ
sets its strength.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 3 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Recap: under- vs over-fitting
Session 02 fitted polynomials to sin(2πx): low degree underfits, high degree
overfits the noise (fit_vs_degree).
Session 01 split the data into train and validation: the fit we like is the one that
does well on data it did not see.
Today we make this precise: measure the trade-off, and steer it with regularization
and principled model selection.
Key idea
More flexibility always lowers training error. The art is keeping generalization error low. That
tension is the whole lecture.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 4 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Train and validation sets
Never grade a model on the data it studied. Split the examples in two:
Training set — the model fits here (it tunes its weights to match these points).
Validation set — held out while fitting; we only judge the model on it, as a stand-in for fresh,
unseen data.
Split the data: fit on train, judge on validation
1.5 true pattern
fit (on train only)
1.0 trainvalidationfit hereheld out, judge here
0.5
y 0.0
0.5
1.0
1.5
0.0 0.2 0.4 0.6 0.8 1.0
x
Intuition
Like an exam: drilling the exact exam questions (memorising) looks perfect, but what counts is doing
well on new questions — that is the validation set.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 5 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Reading over/underfitting from the two errors
Fit on train, then measure the error on both sets — the comparison is the diagnosis:
underfit (degree 1) good (degree 4) overfit (degree 12)
1.5 trainvalidation
1.0
0.5
y 0.0
0.5
1.0
1.5 train RMSE 0.38 train RMSE 0.08 train RMSE 0.03
2.0 valid RMSE 0.70 valid RMSE 0.16 valid RMSE 7.69
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
x x x
Underfit (left): both errors high — too simple to catch the pattern.
Overfit (right): train error tiny, validation error large — a big gap means the model memorised
the training noise.
Good (middle): both low and close together — the target.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 6 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The complexity curve
100 Training error falls monotonically
underfit good overfit with complexity.
scale) Validation error is U-shaped:
(log underfit on the left, overfit on the
error right.
10 1
The sweet spot is the minimum ofsquared the U.
mean training error
validation / test error Intuition
sweet spot A low training error is no proof of a
0 2 4 6 8 10 12 14 good model — it can simply mean
model complexity (polynomial degree)
the model memorised the data.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 7 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The thought experiment: resample & refit
We only ever see one training set — but it is a random draw from a process y = f (x) + ε.
Imagine drawing many fresh training sets — each a new sample of points from the same
process — and refitting the same model family on each. Different data →a different fit (panels
1–3); collect them all and average (panel 4).
training set 1 training set 2 training set 3 collect the fits:
(its own fit) (its own fit) (its own fit) average + spread
1.5 averagetruth ¯f
1.0 the 40 fits
0.5
y 0.0
0.5
1.0
1.5
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
x x x x
Intuition
The blue average fit ¯f = E[ˆy] summarizes the whole family. Two things can go wrong: ¯f can sit off
the truth (bias), or the fits can spread widely (variance). The next slides make each precise.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 8 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two ways to be wrong
simple model (degree 1) flexible model (degree 9)
high bias, low variance low bias, high variance
2.0
average fit
1.5 true sin(2πx)
individual fits
1.0
0.5
y 0.0
0.5
1.0
1.5
2.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
x x
Many fits of the same model family to independently resampled training sets. Left: a simple model —
fits cluster tightly (low variance) but miss the truth (high bias). Right: a flexible model — fits scatter
wildly (high variance) but average out near the truth (low bias).
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 9 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Naming the two errors: a dartboard
Bias = off-center aim · Variance = scattered throws
truth (bullseye) average of the fits one fit (a retraining)
low bias, low variance low bias, high variance
high bias, low variance high bias, high variance
Each dart is one model retrained on a fresh sample; the bullseye is the truth. Bias = how far the darts’
center (green ×) sits from it; variance = how scattered they are. Simple →tight but off-target; flexible
→centered but sprayed.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 10 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Put real numbers on it: one apartment’s rent
Predict the rent of one apartment (70 m2, 2 rooms; true fair rent €1000). We only ever have
one dataset — but imagine retraining on many (listings from different months). Each gives a
rent prediction for this apartment: €860, €880, €900, €950, €960, . . . , averaging to ¯f = €910.
Retrain on many datasets many rent predictions for one apartment
average ¯f true rent f
spread = variance
bias = -90
800 850 900 950 1000 1050
predicted monthly rent ( )
Bias = ¯f −f = €910 −€1000 = −€90 — on average systematically €90 too low.
Variance = how much those predictions scatter around €910 as the training data changes.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 11 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two failure modes in euros — and a floor
Simple model: tight but off Flexible model: centered but jumpy
(high bias, low variance) (low bias, high variance)
average ¯f true rent f average ¯f true rent f
bias = -138 bias 0
600 700 800 900 1000 1100 1200 1300 1400 600 700 800 900 1000 1100 1200 1300 1400
predicted monthly rent ( ) predicted monthly rent ( )
Simple model (rent ≈neighborhood average): tight but off — high bias, low variance
(underfitting).
Flexible model (a deep tree): predictions swing €700–€1300 with the data — low bias, high
variance (overfitting).
Irreducible noise σ2: apartments identical on every recorded feature still rent €980 vs €1080 (an
unrecorded view, timing) — no model erases it.
Key idea
Underfit →high bias; overfit →high variance; error = bias2+variance+σ2 (made exact next).
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 12 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bias and variance, precisely
Average over the resampled training sets, and write ¯f (x) = E[ˆy(x)] for the average fit. Then,
at each input x:
Bias(x) = ¯f (x) −f (x) — how far the average fit sits from the truth f ; a systematic error the
model family cannot shake off.
Variance(x) = E (ˆy(x) −¯f (x))2 — how much a single fit wobbles around that average as the
training set changes.
Irreducible σ2 — the variance of the noise ε in y = f (x) + ε itself, present even for a perfect
model.
Intuition
On the dartboard: ¯f is the darts’ center, bias is center-to-bullseye, variance is the spread of the
darts, and σ2 is the bullseye jittering on its own.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 13 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The bias–variance equation
Put the three pieces together: for squared-error loss the expected error on a fresh point splits
exactly into three parts.
E (y −ˆy)2 = (¯f −f )2 + E[(ˆy −¯f )2] + σ2 .
| {z } | variance{z } noise|{z} bias2
bias2: the average fit ¯f is systematically off the truth (model too simple). Ours to reduce.
variance: the fit wobbles around ¯f from sample to sample (model too flexible). Ours to reduce.
noise σ2: randomness in y itself — a floor no model beats.
Key idea
You cannot zero both — lowering one usually raises the other. The U-shaped validation curve is
bias2 + variance; model selection finds their balance. (The next three slides derive it — optional.)
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 14 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Optional derivation — the expectation toolbox
Now we derive the equation above. Here E[·] averages over the randomness — the training
set (which fixes ˆy) and the noise ε. Three elementary facts do all the work:
1 Linearity — constants pull out and sums split: E[aX + bY ] = a E[X] + b E[Y ], and E[c] = c for a
constant c.
2 Variance = mean squared spread: Var(Z) = E (Z −E[Z])2 ≥0; note that E[Z −E[Z]] = 0.
3 Independence splits a product: if X ⊥Y then E[XY ] = E[X] E[Y ], so a mean-zero, independent
factor kills the term: E[X] = 0 ⇒E[XY ] = 0.
Key idea
Fact 3 is the whole trick: every cross term in the expansion carries a mean-zero, independent factor,
so it vanishes.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 15 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Optional derivation — setup & key identity
At a fixed test point x: the target is y = f + ε — true value f = f (x) fixed; noise with E[ε] = 0,
Var(ε) = σ2, independent of the training data. The fit ˆy comes from the random training set,
and ¯f := E[ˆy] is its average.
Key identity — for a constant a and any random Z:
E (a −Z)2 = a −E[Z] 2 + Var(Z)
Proof. Write a −Z = (a −E[Z]) + (E[Z] −Z) and square. The cross term is
2 (a −E[Z]) E[E[Z] −Z] = 0 (by fact 1, E[E[Z] −Z] = 0), leaving
(a −E[Z])2 + E[(Z −E[Z])2]. □
Intuition
Squared distance from a fixed target a to a random Z = (distance to Z’s mean)2 + Z’s own spread.
We apply it with a = f , Z = ˆy.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 16 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Optional derivation — putting it together
Step 1 — separate the noise. Expand with y = f + ε; the cross term 2 E[(f −ˆy)ε] = 0 (fact
3: ε⊥ˆy, mean 0) and E[ε2] = Var(ε) = σ2, so
E[(y −ˆy)2] = E[(f −ˆy)2] + σ2.
Step 2 — apply the identity with a = f , Z = ˆy (hence E[Z] = ¯f ):
E[(f −ˆy)2] = (f −¯f )2 + Var(ˆy) .
| bias2{z } |variance{z }
Key idea
Combining the two steps gives E[(y −ˆy)2] = bias2 + variance + σ2 — exactly the equation we
stated, now proved.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 17 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The U-curve, dissected
Bias2 falls with complexity;
variance rises.
The U-shaped error splits into bias, variance, and noise
0.40
bias2 σ2 irreducible Their sum + σ2 is the U; its total = bias2 variance + var + σ2 overfit 0.35
(high variance) valley is the best complexity
0.30 (here degree 3).
0.25
The very U we saw in the
0.20contribution validation curve — now
error 0.15 underfit explained.
(high bias)
0.10
best complexity
(minimum of the U) Going deeper (optional)
0.05 How the curves are measured.
0.00 0 2 4 6 8 10 12 14 Monte-Carlo: resample many train-
model complexity (polynomial degree) ing sets per degree; bias2 = mean
(¯f −f )2, variance = mean spread
of the fits.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 18 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why are large weights a problem?
The overfitting side of the U-curve is high variance: the fit swings wildly with the data — and
that wildness comes from large weights.
same data, two fits and their weights
wiggly weights reach ∼109 1.5 truthdata
smooth (deg 3) 109
1.0 wiggly (deg 12) 107
0.5 scale) 105
y 0.0 (log
103
0.5 |wj|
1.0 101
1.5 10 1 wigglysmooth(deg(deg12)3)
0.0 0.2 0.4 0.6 0.8 1.0 0 1 2 3 4 5 6 7 8 9 10 11 12
x coefficient wj (power of x)
To thread every noisy point, the curve bends violently — which needs huge coefficients (big +/−
terms nearly cancelling): the Session 02 “exploding weights”, here |wj| ∼109.
So large weights are the fingerprint of overfitting: keep them small ⇒the fit cannot wiggle as
much ⇒lower variance.
Key idea
That is the idea of regularization: directly discourage large weights to slide back down the U-curve
— without dropping features or lowering the degree.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 19 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The idea: penalize large weights
Instead of restricting the model family, add a penalty that discourages large coefficients:
Jλ(w) = ∥y −Xw∥22 +λ Ω(w) .
| fit the{zdata } stay| {zsimple}
λ ≥0 sets the strength: λ = 0 is plain least squares; large λ forces weights toward 0.
Trading a little bias for a large drop in variance.
Intuition
Same model family, gentler knobs: the penalty slides us down the overfitting side of the
U-curve.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 20 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Ridge (ℓ2) and lasso (ℓ1)
Same fit term, two ways to measure “large weights”:
Ωridge(w) = ∥w∥22 = Pj w j2 , Ωlasso(w) = ∥w∥1 = Pj |wj|.
Ridge is smooth ⇒the closed form from Session 02 (just add λ to the diagonal):
w⋆λ = (XTX + λ I)−1XTy
Intuition
Ridge is a volume knob: it turns every weight down, but never fully off. Lasso adds a mute
switch — the absolute-value penalty snaps small weights to exactly 0 (sparsity). Why, and
how to compute it: next.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 21 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Ridge vs lasso: shrink, or shrink-and-select
Ridge (ℓ2) — pure shrink Lasso (ℓ1) — shrink + select
Pulls all weights down smoothly; keeps every Sets many weights exactly to 0 — a sparse,
feature (none exactly 0). interpretable subset.
Correlated features: shares the weight Correlated features: tends to keep one and
among them — a stable, spread-out estimate. drop the rest (the choice can be unstable).
Smooth ⇒closed form; a safe default. Best when you believe only a few features
truly matter.
Intuition
Rule of thumb: ridge when many small effects add up; lasso when a handful of features do the work.
Want both? Elastic net = λ1∥w∥1 + λ2∥w∥22 shrinks and selects while sharing correlated features.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 22 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why ℓ1 is sparse: the geometry
Loss contours are ellipses around
4 L2 (ridge): ∥β∥2 t 4 L1 (lasso): ∥β∥1 t
the OLS optimum ˆβ.
3 3
ˆβOLS ˆβOLS Penalty as a budget ∥w∥q ≤t:
2 2 shrink until the ellipse first touches
β2 1 1 the region.
ˆβlasso on axis: β1 = 0 ˆβridge: both βj 0
0 0 ℓ2 disk: contact is generic — both
1 1 coordinates nonzero.
2 2
2 1 0 1 2 3 2 1 0 1 2 3 ℓ1 diamond: pointy corners sit on
β1 β1 the axes — contact often lands
there, zeroing a coordinate.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 23 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 24

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
How they shrink: the mechanism behind the corners
For uncorrelated features the penalty decouples: each weight solves the 1-D problem
minw(w −a)2 + λ Ω(w), pulling it from its least-squares value a ≡wOLS toward 0.
Ridge scales every weight; lasso clips small ones to zero
3
no penalty (w = wOLS)
ridge: w = wOLS/(1 + λ)
lasso: soft-threshold at γ Ridge (Ω= w 2): a proportional shrink 2
a
small weights w = — never exactly 0.
w 1 → exactly 0 1 + λ
Lasso (Ω= |w|): the soft-threshold
w = sign(a) max(|a| −γ, 0) — a dead zone coefficient 0
|a| ≤γ that snaps small weights to 0.
dead zone 2γ
penalized 1 Key idea
The kink is the diamond’s corner: near 0 the
2 pull of |w| is constant, so it overpowers small
|wOLS| γ weights (γ grows with λ).
3
3 2 1 0 1 2 3
least-squares coefficient wOLS
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 24 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 25

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
How each is optimized
Ridge is smooth and convex: set the gradient to zero →the closed form
(XTX + λI)−1XTy. The λI also makes the matrix invertible even when features are
collinear. (For big data, plain gradient descent works too.)
Lasso is convex but non-smooth at 0: |w| has a kink, so there is no closed form. The
gradient is replaced by a subgradient (any slope in [−1, 1] at w = 0).
The workhorse is coordinate descent: sweep over the weights, updating one at a time
with the soft-threshold from the last slide — a closed form per coordinate. Fast, and the
exact zeros drop out on their own.
LARS instead traces the whole path (piecewise-linear in λ) in a single pass — exactly how
the coefficient-path plots are computed.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 25 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 26

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Coefficient paths as λ grows
Ridge (L2): smooth shrinkage Lasso (L1): exact zeros
3 weights
w1 w5
w2 w6
2 w3 w7
w4 w8
1 value
0 coefficient
1
2
10 2 10 1 100 101 102 103 104 10 3 10 2 10 1 100 101
penalty strength λ (log scale) penalty strength λ (log scale)
Each curve is one coefficient as the penalty strength λ grows (log axis; scikit-learn calls this parameter
alpha). Ridge: all weights shrink smoothly, none hits zero. Lasso: weights are driven to exactly zero
one by one — the signature of feature selection.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 26 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 27

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
We still must pick λ (or the degree)
A single train/validation split (Session 01) is noisy: the “best” λ depends on which
points happened to land in validation.
Worse with little data — every held-out point is precious.
Intuition
K-fold cross-validation: rotate the validation block so every point is held out exactly once,
then average. A more honest, lower-variance estimate of generalization error.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 27 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 28

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
One split is a lottery
Each thin curve is one random
One split is a lottery; averaging folds is dependable train/validation split. 0.9
12 single 50/50 splits
5-fold CV mean (stable)
0.8 Their “best λ” (the rug at the
bottom) scatters across more
0.7
than a decade.
0.6RMSE
The 5-fold CV mean (thick) is
0.5 smooth, and its minimum isvalidation
0.4 stable.
0.3 Intuition
each split's "best λ" (scattered over a decade) Averaging over folds turns one 0.2
10 4 10 3 10 2 10 1 100 101 102 lucky-or-unlucky guess into a de-
ridge penalty λ (log scale) pendable estimate — which is
exactly why we cross-validate.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 28 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 29

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
K-fold cross-validation
5-fold cross-validation Split the data into K equal folds
(here K = 5).
fold 1 val train train train train
Each fold takes a turn as validation;
fold 2 train val train train train train on the other K −1.
fold 3 train train val train train Average the K scores for one λ;
repeat over a grid of λ.
fold 4 train train train val train
Watch out
fold 5 train train train train val Never let the test set touch model se-
train validation lection. Tune on CV; report once on a
final held-out set.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 29 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 30

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The CV curve picks λ
Choosing λ by cross-validation CV error vs. λ (log axis), with ±
overfit minimum (λ ≈1.9e −01) underfit one standard error across folds.
(λ too small) CV error (mean ± 1 SE) (λ too large) 0.4
(MSE) Same U-shape: too little penalty
0.3 overfits, too much underfits.error
Pick the λ at the valley.
0.2
Going deeper (optional)
0.1 The one-standard-error rule:cross-validation
among λ within one SE of the best,
0.0
10 3 10 2 10 1 100 101 102 103 take the strongest penalty — a
regularization penalty λ simpler model for free.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 30 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 31

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Model selection, animated
Sweeping polynomial degree 0 →18. Left: the fit goes underfit →good →overfit. Right: train RMSE keeps
falling while validation RMSE traces the U; the green star marks the selected degree. (Animated PDF — press
play in a viewer that supports it.)
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 31 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 32

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The selected model
Selected fit, degree 8 Model selection by validation error
1.5 sin(2πx) validation 1.0 train RMSE best (d=8)
train degree-8 fit validation RMSE current
1.0
0.8
0.5
0.6
y 0.0 RMSE
0.5 0.4
1.0
0.2
1.5
0.0
0.0 0.2 0.4 0.6 0.8 1.0 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5
x polynomial degree
The static summary: the fit at the chosen degree (left) and the full train/validation error curves with the
minimum-validation degree marked (right). This is the bias–variance trade-off, resolved by data.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 32 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 33

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Real-world example: ridge & lasso on diabetes
Ridge shrinks; lasso selects (zeros: age, s4, s6)
ridge (L2)
30 lasso (L1)
20
10 coefficient
0
10 standardised
20 test RMSE
OLS 56.4
30 ridgelasso 56.656.5
age sex bmi bp s1 s2 s3 s4 s5 s6
Same data: scikit-learn diabetes, 442 patients, 10 standardized features, 75/25 split; λ chosen by
CV.
Comparable test RMSE — OLS 56.4, ridge 56.5, lasso 56.6.
Ridge shrinks but keeps every feature; lasso sets age, s4, s6 exactly to zero — built-in feature
selection.
Equal accuracy ⇒prefer the simpler, sparse model. Runnable in the companion notebook
session-03-regularization.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 33 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 34

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Summary
Key idea
Training error ̸= generalization error: the validation curve is U-shaped.
That U is bias2 + variance (+ irreducible noise); flexibility trades one for the other.
Regularization adds a penalty λ Ω(w): ridge (ℓ2) shrinks smoothly, lasso (ℓ1) sets
weights to zero (sparsity).
Geometry: the ℓ1 diamond’s corners explain sparsity.
Cross-validation chooses λ (or the degree) honestly; never tune on the test set.
Next: classification — predicting categories instead of numbers.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 34 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 35

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
References
C. M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. (§1.1 polynomial
overfitting; §3.1–3.2 ridge & the bias–variance decomposition.)
T. Hastie, R. Tibshirani, J. Friedman (2009). The Elements of Statistical Learning, 2nd ed.
Springer. (§3.4 shrinkage; §7 model assessment & cross-validation.)
K. P. Murphy (2022). Probabilistic Machine Learning: An Introduction. MIT Press. (Ch. 11
regularization; Ch. 4 model selection.)
G. James, D. Witten, T. Hastie, R. Tibshirani (2013). An Introduction to Statistical Learning.
Springer. (§5 resampling; §6.2 ridge & lasso.)
R. Tibshirani (1996). Regression shrinkage and selection via the lasso. J. R. Stat. Soc. B 58(1),
267–288.
Jan Hurtado (UTEC) Regularization & Model Selection August 23, 2026 35 / 36
```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


#### Página física 36

**Transcripción técnica de la diapositiva (extracción de texto):**

```text

```

**Visual:** curvas de entrenamiento/validación y zonas resaltadas ilustran sesgo-varianza y selección de complejidad; leer los ejes y leyendas antes de comparar.


### 4. ML_Course_Dimensionality_Reduction_Final.pdf — 106 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 11 — Dimensionality
Reduction
PCA, LDA, factor analysis, NMF, and the visu-
alization methods
Jan Hurtado
UTEC
September 25, 2026
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Roadmap
1 Why fewer dimensions 10 Factor analysis
2 PCA geometry 11 Non-negative matrix factorization
3 Eigenvectors and eigenvalues 12 The limit and manifold learning
4 Variance and the covariance matrix 13 t-SNE in depth
5 PCA math 14 UMAP
6 The SVD 15 Beyond PCA: kernel PCA and ICA
7 How many components 16 Practicalities
8 What PCA buys you 17 Wrap-up
9 LDA: using the labels
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 2 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Key terms (for newcomers)
Dimensionality reduction: describing data with Projection: the low-dimensional coordinates of a
fewer numbers while keeping what matters. point on the kept components.
Curse of dimensionality: many features make data Reconstruction: rebuilding the original from the
sparse, distances less meaningful, models overfit. kept components (lossy).
Principal component: an orthogonal direction; SVD: the factorization ˜X = UΣV T; the σk are
PC1 is the direction of maximum variance. singular values.
Eigenvector / eigenvalue: a direction a matrix Scree plot: variance vs component number, used
only stretches, Av = λv, and its stretch λ. to choose how many to keep.
Variance explained: how much of the data’s Manifold: a curved low-dimensional surface the
spread a component captures. data lives on inside a high-D space.
Explained-variance ratio: each component’s share t-SNE / UMAP: nonlinear methods for visualizing
of the total variance. structure in 2-D.
Kernel PCA: nonlinear PCA via the kernel trick Perplexity: t-SNE’s 2H knob = effective number
(Lecture 07). of neighbors.
NMF: additive, parts-based factorization ICA: finds statistically independent sources (blind
(X ≈WH, all ≥0). source separation).
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 3 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Compressing columns, not rows
Clustering (Lec. 10): group the rows
Session 10’s clustering grouped the rows — which
columns = features
points belong together.
Dimensionality reduction is the partner idea: it group 1
compresses the columns — which combinations of points
features carry the signal. = group 2
rows
Still unsupervised: just {xi} with d features and no group 3
target y; we want a shorter description of each point.
Why bother? Visualization, compression,
Dim. reduction: compress the columns denoising, dropping redundant features, and speed.
d features k ≪d
PCA
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 4 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why reduce? Five payoffs
Why reduce dimensions? Five payoffs
Visualize Compress Denoise Drop redundancy Speed up
(2-3 D picture) (fewer numbers) (drop the tail) (1 axis, not 2) (fewer inputs)
d noisy
recovered
2 time
feature train
faster∼5×
k ≪d
store d store k feature 1 d feats k PCs
Visualization (2–3 D plots), compression (store k ≪d numbers), denoising (drop the low-variance
tail), dropping redundant features, and speed (fewer inputs to every downstream model).
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 5 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The curse of dimensionality
Distances lose meaning
Data becomes sparse: to keep the same density 1.0
you need exponentially more points as d grows.
0.8
Distances flatten — nearest and farthest farthest/ 0.6 near & far become
equidistant neighbors end up almost equidistant, so “closest distance 0.4 point” loses its meaning. nearest
0.2
Volume concentrates in the corners; low-D 0.0
geometric intuition simply fails. 0 10 20dimension30d 40 50
More features means more ways to overfit noise. Volume flees to the corners
100
10 1
/ 10 2
3 10 ballvolume 10
5 10 inscribedcube 4
6 10
10 7
2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0
dimension d
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 6 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The geometry of the curse
The curse of dimensionality is geometric
2-D: circle inscribed in a square Inscribed ball vanishes: volume flees to the corners
1.0
d = 2: 79%
0.8
d = 3: 52%
ball 0.6
78.5% Vball/Vcube
0.4
d = 5: 16%
0.2
corners d = 10: 0.25%
0.0
2 4 6 8 10 12
dimension d
The circle fills most of the square, but the inscribed ball’s share of the cube’s volume →0 as d grows —
almost all the volume ends up in the corners, far from the center. High-D volume, distance, and density
all defy low-D intuition.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 7 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The same points, ever-emptier space
The same 27 points, in ever-emptier space
1-D: packed together 2-D: spreading out 3-D: mostly empty
1
1
0
1
0
0 1 0
0 1 0 1
Hold the number of points fixed: elbow-to-elbow in 1-D, spreading out in 2-D, marooned in
mostly-empty space by 3-D — and it only gets worse as d climbs.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 8 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Not all features pull their weight
Not all features pull their weight
(a) Relevance to wear (target) (b) Redundancy (|feature-feature corr|) 1.0 A machine is monitored by five sensor columns — temp, 1.0 1.00 1.00 0.05 0.02 0.06 signalredundantnoise / constant temp 0.75 0.8
0.64 0.64 temp_F 1.00 1.00 0.05 0.02 0.06 temp_F, vibration, noise, constant — and we log its wear)|0.6 0.8
vibration 0.05 0.05 1.00 0.03 0.05 wear as the target to gauge each column against. 0.4 0.6 0.4 |corr(feature, noise 0.02 0.02 0.03 1.00 0.00
constant 0.06 0.06 0.05 0.00 1.00 Signal: temp and vibration track the wear. 0.20.0 0.01 0.01 0.2
temp temp_F vibration noise constant temptemp_Fvibration noiseconstant 0.0
Redundant: temp_F is just temp in Fahrenheit — the (c) Informative feature: clear trend (d) Useless feature: no trend
same information stored twice. 100 100
90 90
Useless: noise is unrelated to wear; constant barely wear 8070 wear 8070
varies. 6050 6050
40 40
20 30 40vibration50(signal)60 70 80 3 2 noise1 (irrelevant)0 1 2 3
(full view next slide)
Intuition
The wear (target) is only a yardstick to see that columns differ — PCA itself is unsupervised: it finds
a few informative combinations of features and discards the redundant/noisy rest, using no target at
all.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 9 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Some features matter, others do not
Not all features pull their weight
1.0 (a) Relevance to wear (target) (b) Redundancy (|feature-feature corr|) 1.0
signal
redundant temp 1.00 1.00 0.05 0.02 0.06
0.8 0.75 noise / constant 0.8
wear)|0.6 0.64 0.64 temp_F 1.00 1.00 0.05 0.02 0.06 0.6
vibration 0.05 0.05 1.00 0.03 0.05
0.4 0.4 |corr(feature, noise 0.02 0.02 0.03 1.00 0.00
0.2 0.2
0.01 0.01 constant 0.06 0.06 0.05 0.00 1.00
0.0 0.0
temp temp_F vibration noise constant temptemp_Fvibration noiseconstant
(c) Informative feature: clear trend (d) Useless feature: no trend
100 100
90 90
80 80
wear 70 wear 70
60 60
50 50
40 40
20 30 40 50 60 70 80 3 2 1 0 1 2 3
vibration (signal) noise (irrelevant)
(a) relevance to wear (the target), (b) redundancy (temp=temp_F), (c) an informative feature’s trend
vs. (d) a useless feature’s shapeless cloud.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 10 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
PCA: find the directions that matter
Principal components = orthogonal axes of maximum variance
Principal Component Analysis finds ordered mean
orthogonal directions along which the data varies 4
most.
PC1 = direction of maximum variance; PC2 = the 2
next-most-variable direction orthogonal to PC1; and so x2
on. 0
PC2
Keep the top few and project: a d-dimensional point
becomes a k-dimensional summary, k ≪d. 2
PC1
The directions are new axes, aligned to the data’s own
shape. 4 2 0 2 4 6
x1
(full view next slide)
Intuition
Imagine the cloud as a rugby ball. PC1 points along its longest axis, PC2 across the next-widest.
Describe each point by where it sits along those axes and you have captured most of the ball with
far fewer numbers.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 11 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Principal directions of a cloud
Principal components = orthogonal axes of maximum variance
mean
4
2
x2
0
PC2
2
PC1
4 2 0 2 4 6
x1
PC1 (long arrow) and PC2 (short arrow) emanate from the mean, along the directions of greatest and
next-greatest spread.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 12 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two equivalent views of the same fit
Maximize projected variance: the direction the points spread along most.
Minimize reconstruction error: the line whose perpendicular residuals are smallest.
Same answer (total spread is fixed) — and not least-squares regression: PCA uses perpendicular,
not vertical, distances.
Minimize reconstruction error The 1-D summary
(gray = residuals) (best rank-1 reconstruction)
PC1 line
x2 x2
x1 x1
Keeping PC1: each point drops onto the line; the leftover perpendicular segments are the reconstruction
error PCA makes smallest.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 13 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which direction wins? Watch the variance
A candidate direction sweeps around the cloud; the projected variance peaks at PC1. (Animated PDF
— press play to view.)
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 14 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Variance as a function of angle
PCA picks the direction of maximum projected variance
4.5
PC1 direction
4.0
3.5
3.0 variance 2.5
2.0
1.5 projected
1.0
0.5
0 20 40 60 80 100 120 140 160 180
direction angle (degrees)
The projected variance traced over every direction: a single clear peak marks PC1, a trough marks the
orthogonal PC2.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 15 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Another view: the projected distribution
Projected variance = spread of the 1-D shadow
“Variance” is just the spread of the 1-D
The cloud and its two axes
shadow — the histogram of the projected
values.
Project onto PC1: the shadow is wide (large
variance). PC2
Project onto PC2: the shadow is narrow PC1
(small variance).
PCA picks the direction whose projected
distribution is widest. PC1's shadow widest, PC2's narrowest
onto PC1 (variance 4.5)
onto PC2 (variance 0.3)
6 4 2 0 2 4 6
projected coordinate
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 16 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Watch the projected distribution breathe
As the direction rotates, the 1-D projected distribution stretches (toward PC1) and squeezes (toward
PC2). (Animated PDF — press play.)
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 17 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Eigenvectors: the directions a matrix only stretches
Eigenvectors: the directions A only stretches (Av = λv)
A matrix A acts by multiplication: v 7→Av. For most
directions it both stretches and rotates.
An eigenvector is a direction A only stretches — it Av (rotated)
does not rotate it: Av = λv.
The eigenvalue λ is the stretch factor (> 1 grow, < 1 v
shrink, < 0 flip).
v2: λ2 = 0.85
A symmetric matrix has orthogonal eigenvectors and
real eigenvalues — exactly what we need for the v1: λ1 = 2.55
covariance matrix. unit circle
image ellipse (Av)
(full view next slide)
Intuition
Eigenvectors are the “natural axes” of the matrix: the special directions it leaves pointing the same
way, merely scaled.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 18 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Av traces an ellipse; eigenvectors are its axes
Eigenvectors: the directions A only stretches (Av = λv)
Av (rotated)
v
v2: λ2 = 0.85
v1: λ1 = 2.55
unit circle
image ellipse (Av)
As v runs over the unit circle, Av sweeps an ellipse. A generic v is rotated; only along the eigenvectors
(the ellipse axes) is Av = λv.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 19 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where the special directions come from
Eigenvectors are the directions a matrix only stretches
Diagonal Λ: the axes are eigenvectors Symmetric A = VΛV T: rotated axes are eigenvectors
unit circle
image ellipse
green = eigenvectors (only stretched); blue v → orange Av (a generic vector is turned)
A diagonal matrix stretches the standard axes (left); a symmetric A = V ΛV T stretches a rotated pair
(right) — its eigenvectors (columns of V ). This is exactly why PCA works on the covariance matrix.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 20 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Watch: only eigenvectors don’t rotate
The angle between v and Av drops to zero exactly at the two eigenvector directions. (Animated PDF
— press play.)
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 21 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Computing them: the characteristic equation
Av = λv ⇔(A −λI)v = 0, which needs det(A −λI) = 0.
Going deeper (optional)
For A = ( 21 12 ):
det(A −λI) = (2 −λ)2 −1 = (λ −1)(λ −3) = 0 ⇒λ1 = 3, λ2 = 1.
Back-substitute: λ1 = 3 ⇒v1 ∝(1, 1); λ2 = 1 ⇒v2 ∝(1, −1) — orthogonal, as promised.
Intuition
For big matrices nobody solves the polynomial — iterative methods (power iteration) or the SVD
find the eigenpairs stably.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 22 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
You have used these before — from the other end
Session 10 ended by pointing here, and it is worth collecting the debt:
spectral clustering (session 10) PCA (here)
the matrix the graph Laplacian L the covariance C
which eigenvectors the smallest the largest
what they mean directions of least disagreement directions of greatest spread
what you do with them cut the graph project the data
Session 10 used the Fiedler vector — the eigenvector of the second smallest eigenvalue — to split
a graph in two.
PCA takes the eigenvectors of the largest eigenvalues to keep as much variance as possible.
Key idea
Same machinery, opposite end of the spectrum. An eigen-decomposition is not a technique belonging
to one method — it is how you ask a symmetric matrix which directions it treats specially, and what
“specially” means is up to you.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 23 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 24

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Variance and covariance: the tools of PCA
Variance of feature j — its spread about the mean over the n points: σ2j = n1 Pi(xij −µj)2.
Covariance of features j, k — how they move together: σjk = n1 Pi(xij −µj)(xik −µk).
Same per-feature variances but different covariance ⇒very different clouds (round vs. tilted).
These numbers are exactly the entries of the covariance matrix C built next.
Variance and covariance: the raw material of PCA
Variance: spread about the mean Uncorrelated: σ12 = 0 Correlated: σ12 > 0
µ
2σ x2 x2
4 2 0 2 4 x1 x1
x
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 24 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 25

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The covariance matrix is the shape of the data
Collect those numbers into the covariance matrix C — symmetric, so its eigenvectors are orthogonal.
Its eigendecomposition names the pieces PCA needs:
σ21 σ12 C = = V ΛV T, V = [ v1 v2 ], Λ = diag(λ1, λ2).
σ12 σ22
The covariance matrix C shapes the Gaussian (and PCA reads off its eigenvectors)
Isotropic No correlation With correlation
C = σ2I C = diag(σ21, σ22) = Λ C = VΛV T
eigenvectors = axes, eigenvalues = variances
C turns a round cloud into the shape we see. PCA reads its eigenvectors (columns of V ) off as the
axes, its eigenvalues λk as the variances along them.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 25 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 26

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Deriving PCA, 1: projected variance
Principal components = orthogonal axes of maximum variance
mean Center the data, ˜xi = xi −µ, so n1 Pi ˜xi = 0.
Pick a unit direction u with ∥u∥= 1. A point’s coordinate 4
along it is the projection
2
⟨˜xi, u⟩ x2
= uT˜xi. Proju(˜xi) = 0 ∥u∥
PC2
PCA asks: which u makes those projections most spread 2
out? PC1
The projections are already centered, so their variance keeps 4 2 0 x1 2 4 6
only the squared term: a direction u; project every point onto it
Var uT˜xi = 1 X uT˜xi 2.
n
i
Intuition
The projections’ mean is uT0 = 0, so “spread” is just the average squared projection.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 26 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 27

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Deriving PCA, 2: the covariance matrix
Write the squared projection as a quadratic form in u (it is a scalar, so it equals its own
transpose):
uT˜xi 2 = uT˜xi ˜xTi u = uT ˜xi ˜xTi u.
Average over the data and pull the constant u outside the sum:
Var uT˜xi = uT n1 Pi ˜xi ˜xTi u = uTC u, C = n1 ˜X T ˜X.
Key idea
The projected variance in any direction is uTC u — the covariance matrix C is exactly what turns
a direction into a variance. So PCA becomes one optimization:
u⋆= arg max uTC u.
∥u∥=1
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 27 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 28

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Deriving PCA, 3: the eigenvalue problem
Maximize uTCu subject to uTu = 1 with a Lagrange Eigenvectors of C = principal axes (λk = variance)
multiplier λ: 2σ covariance ellipse
L(u, λ) = uTC u −λ uTu −1 .
Setting the gradient to zero,
∇uL = 2 Cu −2λu = 0 =⇒ Cu = λu .
v2 (λ2 = 0.87)
The optimal directions are eigenvectors of C.
v1 (λ1 = 5.78)
Multiplying Cu = λu by uT gives uTCu = λ: the
captured variance is the eigenvalue. So λ1 →PC1,
λk λ2 →PC2, . . . arrows = eigenvectors of C; length ∝ p
Going deeper (optional)
This is the Rayleigh quotient uTCu/uTu: its maximum is the largest eigenvalue, reached at the top
eigenvector (Bishop, PRML Ch. 12; Jolliffe; Shlens’ PCA tutorial).
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 28 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 29

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The PCA recipe
Having shown Cu = λu, the algorithm is immediate:
Center the data: subtract the mean µ = n1 Pi xi to get ˜X (rows are xi −µ).
Form the covariance matrix C = n1 ˜X T ˜X ∈Rd×d.
Eigendecompose: the eigenvectors are the principal components, the eigenvalues the variances
along them,
C = V ΛV T, Cvk = λkvk, λ1 ≥λ2 ≥· · · ≥0.
Intuition
The eigen-decomposition rotates the axes so the data’s spread lines up with them: each new axis is
a vk, and how far the data stretches along it is λk.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 29 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 30

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Project and reconstruct
Stack the top k eigenvectors as columns of Vk ∈Rd×k.
Projection — the k-dimensional coordinates of a point:
z = VkT (x −µ) ∈Rk.
Reconstruction — the best rebuild from those coordinates:
ˆx = Vk z + µ.
The reconstruction is lossy unless k = d; the error dropped equals Pj>k λj — exactly the variance
in the discarded directions.
Key idea
PCA is a rotate-then-truncate: rotate into the eigenvector axes, keep the k high-variance coordi-
nates, zero the rest. Cheap, exact, and reversible up to the variance you chose to drop.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 30 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 31

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
PCA as matrices — the whole pipeline
PCA as matrices: form C, eigendecompose, then project and reconstruct
1. Covariance
first:
˜X = X −µ C = n1 ˜XT × ˜X center
d × n
d × d n × d
2. Eigendecomposition (this is PCA)
keep top-k
columns C = V Λ V T × × →Vk of V
d × d PCs (cols) λ1 ≥ ≥λd d × d
3. Project (reduce) and reconstruct
kT +µ Z = ˜X × Vk ˆX = Z V
n × k n × d d × k n × d n × k
Form C = n1 ˜X T ˜X; eigendecompose C = V ΛV T (columns of V are the PCs, top k kept); project Z = ˜XVk,
reconstruct ˆX = ZV kT + µ.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 31 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 32

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Standardize first (usually)
Watch out
PCA chases variance, so it is not scale-invariant: a feature in large units (income in dollars, or
temp_F) dominates one in small units for no good reason. Unless features share units, standardize
first (zero mean, unit variance) — equivalently, run PCA on the correlation matrix.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 32 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 33

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The singular value decomposition
Every matrix factors as ˜X = UΣV T with U, V orthogonal and Σ diagonal (singular values
σ1 ≥σ2 ≥· · · ≥0).
Geometrically: rotate (V T) →scale (Σ) →rotate (U) — it turns the unit circle into an ellipse.
Columns of U: left singular vectors; columns of V : right singular vectors; the σk are the ellipse’s
semi-axis lengths.
Intuition
The SVD works for any matrix (not just square/symmetric) and is the master factorization behind
PCA, recommender systems, and low-rank compression.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 33 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 34

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
SVD geometry: rotate, scale, rotate
SVD geometry: every matrix is a rotation, then a scaling, then a rotation
unit circle
(inputs v1, v2) V T: rotate Σ: scale U: rotate →M = UΣV T
V T rotates, Σ scales the circle into an ellipse, U rotates it into place. Singular values = semi-axis
lengths.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 34 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 35

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Watch the circle become an ellipse
Rotate →scale →rotate. (Animated PDF — press play.)
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 35 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 36

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
SVD as blocks — and the best rank-k fit
SVD as matrices: ˜X = UΣV T, truncated to rank k
Economy SVD of the centered data
cols of U: left sing. vecs
diag Σ: σ1 ≥σ2 ≥ ˜X = U × Σ × V T
rows of V T: PCs
d × d d × d
n × d n × d
Keep the top k triplets: best rank-k approximation
store k(n + d + 1)
instead
of nd; and k T ⇒ numbers ˜X ≈ Uk × Σk × V λk = σ2k/n
k × k k × d
n × d n × k
Keep the top k triplets: ˜X ≈UkΣkV kT is the best rank-k approximation (Eckart–Young), storing k(n+d+1)
numbers.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 36 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 37

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why not just eigen-decompose C?
The recipe said “form C = n1 ˜X T ˜X, then eigen-decompose”. Building C first is both wasteful and fragile:
Cost & size. ˜X T ˜X costs O(nd2) and is d × d — for d = 104 features that is 108 entries before
the eigensolve.
Precision. Squaring the data squares the condition number, κ(C) = κ(˜X)2. Small-variance
directions (small σk) drown in rounding error — the fine structure PCA is meant to find.
Rank. If d > n (more features than samples) C is singular (rank ≤n), so the eigen-route is
ill-posed.
Watch out
Wish: read the principal directions straight off ˜X, without ever forming ˜X T ˜X. That is exactly what
the SVD does.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 37 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 38

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The SVD is PCA
Substitute ˜X = UΣV T into the covariance (UTU = I):
C = n1 ˜X T ˜X = nV1 ΣTUTUΣV T = V nΣTΣ1 V T.
Compare with C = V ΛV T: the right singular vectors V are the principal components, and
λk = σ2k/n.
The scores need no C either: Z = ˜XVk = UkΣk.
So libraries compute PCA via the SVD of ˜X — it never forms ˜X T ˜X, avoiding the squared
condition number and reading the PCs off ˜X accurately (even when d > n).
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 38 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 39

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Eigen of C vs. SVD of ˜X
Eigen-decomposition of C SVD of ˜X
needs the square matrix C = n1 ˜X T ˜X works directly on any ˜X
forms ˜X T ˜X: O(nd2), d × d never forms it
squares the condition number keeps κ(˜X) — more stable
breaks when d > n (C singular) fine when d > n
PCs = eigenvectors, var = λk PCs = cols of V , var = σ2k/n
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 39 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 40

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Eigen of C vs. SVD of ˜X
Eigen-decomposition of C SVD of ˜X
needs the square matrix C = n1 ˜X T ˜X works directly on any ˜X
forms ˜X T ˜X: O(nd2), d × d never forms it
squares the condition number keeps κ(˜X) — more stable
breaks when d > n (C singular) fine when d > n
PCs = eigenvectors, var = λk PCs = cols of V , var = σ2k/n
Key idea
PCA and SVD are equivalent for centered data: PCA is the statistical view (maximum variance),
the SVD is the stable, efficient way to compute it.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 39 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 41

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Low-rank compression in action
Low-rank approximation by SVD: a few singular values carry the picture
original r = 1 r = 2 r = 5
(rank 220) 99.2% energy, 110× 99.7% energy, 55× 99.9% energy, 22×
r = 10 r = 25 r = 50 r = 100
99.9% energy, 11× 100.0% energy, 4× 100.0% energy, 2× 100.0% energy, 1×
An image rebuilt from its top r singular values: a few already carry the picture — the same “keep the
top directions” move as PCA.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 40 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 42

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Choosing k: the explained-variance ratio
Scree plot: digits, 64-D, standardized
no elbow --- and reaching 90 percent costs 31 of the 64 components Each component’s share of the total spread is its 1.0
0.12
explained-variance ratio λi/ Pj λj.
0.10 0.8
The scree plot shows these per component; look for
0.6 variance the elbow where extra components stop adding explained 0.08 31 PCs for 90 percent
much. 0.06
0.4 40 PCs for 95 percent cumulative variance 0.04 Or pick k for a cumulative threshold — e.g.
enough to explain 90–95% of the variance. 0.02 0.2
0.00 0.0
0 10 20 30 40 50 60
principal component
(full view next slide)
Intuition
The eigenvalues are a budget of “how much shape” the data has. Spend it on the biggest few
directions and you keep most of the picture with a fraction of the numbers.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 41 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 43

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Scree plot: where to stop
Scree plot: digits, 64-D, standardized
no elbow --- and reaching 90 percent costs 31 of the 64 components
1.0
0.12
0.10 0.8
0.08
0.6 explained 31 PCs for 90 percent variance
0.06
0.4 variance 40 PCs for 95 percent cumulative 0.04
0.2
0.02
0.00 0.0
0 10 20 30 40 50 60
principal component
Per-component variance (bars) falls off fast — and then keeps going. On these standardized digits the
cumulative curve reaches 90% only at the 31st of 64 components and 95% at the 40th. There is no
elbow here, and “a few components carry it” would be wrong.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 42 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 44

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Visualizing high-dimensional data
Handwritten digits projected to 2-D by PCA
9
10.0
8
7.5
7
5.0 6
2.5 5
PC2 digit
4
0.0
3
2.5
2
5.0
1
7.5 0
10.0 7.5 5.0 2.5 0.0 2.5 5.0 7.5
PC1
The 64-D handwritten digits projected to their top 2 PCs: same-digit points cluster and separate — real
structure, though with some overlap.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 43 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 45

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Compression: a digit from k components
PCA compression: a digit rebuilt from its top-k components
k=1 k=2 k=4 k=8 k=16 k=32 k=64
original 15% var 29% var 49% var 67% var 85% var 97% var 100% var
One digit rebuilt from k = 1, 2, 4, 8, 16, 32, 64 components. A few components already give a
recognizable digit; the rest add fine detail.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 44 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 46

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Example: PCA on time-domain signals
PCA on time-domain signals: 2 components capture 85% of the variance
(a) Raw signals (noisy) (b) Top 2 PCs = learned patterns (c) One signal from 2 of 120 numbers
PC1 (68% var) raw
PC2 (18% var) from 2 PCs
0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0 0.0 0.2 0.4 0.6 0.8 1.0
time time time
Each signal is a noisy mix of two hidden waveforms. PCA on the signal matrix (rows = signals, columns
= time) recovers those patterns as its top PCs, so two numbers per signal already reconstruct it — PCA
works directly on raw time series.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 45 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 47

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Example: fewer features, faster learning
Fewer features ⇒ faster training, near-equal accuracy
Training time: 3.7× faster Accuracy: about the same (25 PCs keep 44% var) 0.7 0.68s
1.0 0.95 0.91
0.6
(s) 0.8
0.5
time
0.4 0.6 accuracy
0.3 training test 0.4
0.2 0.18s SVM
0.2
0.1
0.0 0.0
200 raw 25 PCs 200 raw 25 PCs
features features
An SVM trained on the top 25 principal components instead of the 200 raw features trains several times
faster with no loss of test accuracy — PCA as cheap, decorrelating, denoising preprocessing.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 46 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 48

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Example: easier clustering after PCA
Clustering is easier in the reduced space
Two raw features: clusters overlap PCA 2-D: clusters separate
(k-means ARI = 0.45) (k-means ARI = 1.00)
2
feature PC2
raw
raw feature 1 PC1
The same three clusters overlap in two raw features but separate cleanly in the top two PCs, where
k-means recovers them exactly (ARI = 1.0). Reduce first, then cluster.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 47 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 49

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
More than a picture
Key idea
Keep the top-k directions, drop the rest. That single move gives you: visualization (2–3 D plots),
compression (store k numbers instead of d), denoising (noise lives in the low-variance tail you
discard), and whitening (rescale components to unit variance for downstream models).
Intuition
The same idea applied to face images gives eigenfaces — a handful of “prototype faces” whose
weighted sum reconstructs any face in the set.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 48 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 50

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
PCA never looked at y
Everything so far has been unsupervised: PCA Principalmean components = orthogonal axes of maximum variance
maximizes the variance of the projection, and the 4
labels never enter the calculation.
Usually that is the point — there are no labels. 2
But when there are labels and the goal is to x2 0
separate the classes, maximizing variance answers PC2
the wrong question. 2
PC1
“Spread out” and “well separated” are not the
same thing, and the next frame is a case where 4 2 0 2 4 6
they point in perpendicular directions. x1
Key idea
Linear discriminant analysis asks the supervised question instead: which direction separates the
classes best? You have met it before — session 05 used it as a classifier.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 49 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 51

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where variance and separation disagree
Both classes have the SAME spread along x1 --- so the direction of greatest variance carries no class information at all
the data: PC1 explains 99.24 percent of the variance
and runs straight along the classes
5 LDA
x2 0 PC1
5
30 20 10 0 10 20 30
x1
after PCA
best possible accuracy→ 1D 0.5417 best possibleafter LDAaccuracy→ 1D 0.9867
20 classclass 01 30 classclass 01
15
count 10 count 20
10
5
0 0
30 20 10 0 10 20 4 2 0 2 4
the single PCA coordinate the single LDA coordinate
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 50 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 52

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Reading those numbers
PCA →1D LDA →1D
share of the variance kept 99.24% —
best possible accuracy from that one number 0.5417 0.9867
used the labels? no yes
Both classes have the same distribution along the long axis, so the direction carrying 99% of the
variance carries zero class information.
PCA is not doing badly here. It is answering the question it was asked.
Watch out
“I ran PCA first to reduce the noise, then classified” is a standard pipeline and it can quietly cost
you everything. If you have labels and you want separation, say so.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 51 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 53

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two scatters, not one
Split the spread of the data into the part within each class and the part between the classes:
SW = X X (xi −µk)(xi −µk)T how fat each class is
k i: yi=k
SB = X nk (µk −µ)(µk −µ)T how far apart the class means are
k
Project onto a direction w and both become numbers: wTSW w and wTSBw.
LDA maximizes their ratio — far apart, and each one tight:
wTSBw
J(w) =
wTSW w
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 52 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 54

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Solving it
J is a ratio, so scaling w changes nothing — fix wTSW w = 1 and maximize the top. The
Lagrange condition is
SBw = λ SW w a generalized eigenproblem
S−1W SB w = λ w an ordinary one, if SW is invertible
For two classes you do not need an eigensolver at all. SB has rank 1, so SBw always points
along µ1 −µ0, and
w ∝S−1W (µ1 −µ0)
Read it as: take the direction between the means, then correct it for the shape of the classes.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 53 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 55

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A generalized eigenproblem, in words
An ordinary eigenproblem asks: which w does A only Eigenvectors: the directions A only stretches (Av = λv)
stretch?
A generalized one, SBw = λSW w, asks: which w Av (rotated)
does SB stretch in the same way SW does?
v
Equivalently: whiten the data with SW first, then do
an ordinary eigen-decomposition in the whitened
v2: λ2 = 0.85
space.
If SW is singular — which happens whenever there v1: λ1 = 2.55
are more features than samples — the inverse does unitimagecircleellipse (Av)
not exist and you must regularize it (SW + εI). That
is what “shrinkage LDA” means.
Watch out
S−1W is where LDA breaks in practice. With d > n it is not merely inaccurate — it does not exist.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 54 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 56

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
LDA by hand
Each class is a square of side 2, so each contributes 4I and SW = 8I --- which is what makes the inverse trivial
eight points, all integers and the whole of LDA, on paper
10 class 0 µ0 (3, 3)
class 1 w ∝(1, 1) µ1 (7, 7)
8
µ1 −µ0 (4, 4) (7, 7)
6 SW 8I --- rows (8, 0) and (0, 8)
x2 SW−1 0.125I
4
w = SW−1 (µ1 −µ0) (0.5, 0.5) (3, 3)
direction (1, 1)/ 2 2 p
sklearn's LDA returns the same direction, to machine precision.
0
0 2 4 6 8 10
x1
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 55 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 57

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check it yourself
µ0, µ1 (3, 3) and (7, 7)
µ1 −µ0 (4, 4)
each class’s scatter 4I — a square of side 2 about its mean
SW 4I + 4I = 8I
S−1W 0.125 I
w = S−1W (µ1 −µ0) 0.125 ·√(4, 4) = (0.5, 0.5)
direction (1, 1)/ 2
Here SW is a multiple of the identity, so the correction does nothing and w just points between the
means.
That is the special case. When the classes are elongated, S−1W turns the direction away from the
line joining the means — which is exactly what the cigars figure showed.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 56 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 58

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two warnings about SB
definition on the eight points used when
Pk nk(µk −µ)(µk −µ)T 32 in each entry any K
(µ1 −µ0)(µ1 −µ0)T 16 in each entry K = 2 only
Both are called SB. They differ by a constant — here a factor of 2 — so the direction LDA returns
is identical.
But a slide that mixes the two will not add up, and a student checking the arithmetic will think
they are wrong. This deck uses the weighted form throughout.
Watch out
Whenever two textbooks give you different constants for the same matrix, check whether the ratio
you actually maximize is affected. Here it is not.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 57 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 59

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
How many directions? At most K −1
Where LDA stops --- and the second case is exactly the one session 05 handed to QDA
limit 1: at most limit 2: same mean, different SHAPE K −1 directions K SB is built from means about their own average, 2.32, so LDA has almost nothing ∥SB∥= to maximize so one degree of freedom is already spent best 1-D accuracy after LDA: 0.5675
6
K −1 8 5 directions 5 6
both class means LDA 4 4 4 sit here 2
3 3 x2 0 usable
2 2 = 2
4
SB 1 1 6
of
rank 0 2 3 4 5 6 8 5 0 5
number of classes K x1
Left: SB is built from K class means measured about their own average, so one degree of freedom is
already spent and its rank is K −1. Two classes give you exactly one direction — no matter how many
features you started with.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 58 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 60

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What LDA cannot see
Where LDA stops --- and the second case is exactly the one session 05 handed to QDA
limit most directions limit 2: same mean, different SHAPE Right panel: two classes with the same mean
SB is built from 1:Katmeans aboutK −1 their own average, 2.32, so LDA has almost nothing to maximize ∥SB∥= so one degree of freedom is already spent best 1-D accuracy after LDA: 0.5675 and very different shapes. They are obviously 8 K −1 5 6directions 65 separable — and LDA scores 0.5675, near
LDA 4 4 42 bothsit hereclass means
3 x2 3 0 chance.usable
2
= 2 2
4 SB 1 1 6 SB is built from the means. Equal means, no
of
rank 0 2 3 4 5 6 8 5 0 5 SB, nothing to maximize.
number of classes K x1
LDA also assumes the classes share one
covariance — which is exactly session 05’s
pooled Σ.
Intuition
Session 05 met this same wall and stepped over it with QDA: let each class keep its own covariance
and the boundary becomes quadratic. Same assumption, same fix.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 59 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 61

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The same method, twice
session 05 here
what it produced a decision boundary a projection direction
the assumption one pooled Σ for all classes SW , the same pooled scatter
the answer Σ−1(µ1 −µ0) S−1W (µ1 −µ0)
The boundary session 05 drew is perpendicular to the direction this lecture projects onto. They are
the same vector — and that is checkable, not a figure of speech.
On the cigars: session 05’s Σ−1(µ1 −µ0) and the top generalized eigenvector of S−1W SB differ by
0.0000000000◦.
So LDA is not a second method to learn. It is one method you can use to classify or to reduce,
and the arithmetic does not change.
Key idea
Session 05 called it the normal to a decision boundary. Here it is an axis to project onto. One
computation, two uses — which is why the K −1 limit also explains why a two-class LDA classifier
only ever needs one number.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 60 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 62

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
PCA or LDA?
PCA LDA
uses labels no yes
maximizes variance of the projection between/within scatter ratio
components available up to d at most K −1
needs nothing invertible SW , so n > d or shrinkage
assumes nothing classes share a covariance
typical use compression, denoising, plotting a supervised pre-projection
Key idea
Use PCA when you have no labels or want to compress. Use LDA when you have labels, want
separation, and K is large enough that K −1 directions is enough. They are not rivals — running
PCA first and LDA second is a standard way to make SW invertible.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 61 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 63

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A different question about the same matrix
No correlation With correlation PCA asks a geometric question: which directions IsotropicThe covariance matrix C shapes the Gaussian (and PCA reads off its eigenvectors)
C C = σ2I C = diag(σ2 1, σ22) = Λ
carry the most variance? = VΛV T
Factor analysis asks a generative one: what small
set of hidden causes would produce data that looks
like this? eigenvectors = axes, eigenvalues = variances
The model:
x = W z + µ + ε
with z ∼N(0, I) the latent factors, W the
loadings, and ε ∼N(0, Ψ) with Ψ diagonal.
Intuition
This is session 05’s habit of mind — write down how the data was generated — applied to dimen-
sionality reduction. And like session 10’s mixtures, it is fitted by EM.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 62 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 64

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What the model says about the covariance
Take the covariance of x = W z + µ + ε, with z and ε independent:
Cov(x) = W Cov(z) W T + Cov(ε)
= WW T + Ψ since Cov(z) = I
So the model splits the covariance in two: a low-rank part WW T that the k shared factors
explain, and a diagonal part Ψ that is private to each feature.
That diagonal is the whole difference from PCA. PCA has no noise term at all; the closest
probabilistic version, probabilistic PCA, forces Ψ = σ2I — the same noise in every feature.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 63 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 65

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why the diagonal matters
One feature made ten times noisier. PCA cannot say "that one is just noisier"; Ψ is exactly that sentence.
PCA factor analysis
direction moves 8.280 degrees direction moves 0.003 degrees a factor of 2,782
0.7 original 0.7 original 8.280
0.6 after x5 × 10 0.6 after x5 × 10 moved 101
0.5 0.5 100 vector) vector)
10 1 (unit 0.4 (unit 0.4 direction
0.3 0.3 the 10 2
0.2 0.2 0.003 |loading| |loading| 10 3 0.1 0.1 degrees
0.0 0.0 10 4
x1 x2 x3 x4 x5 x1 x2 x3 x4 x5 PCA FA
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 64 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 66

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Reading that experiment
Five features driven by one hidden factor. Then feature 5 is multiplied by ten — the same information,
recorded on a noisier instrument.
PCA’s loading direction moves 8.280◦
factor analysis’s moves 0.003◦
ratio about 2800×
PCA maximizes variance, and feature 5 now has a hundred times more of it. It has no vocabulary
for “that one is just noisier”.
Ψ is that vocabulary: factor analysis absorbs the extra variance into that feature’s private noise
and leaves the shared factor alone.
Key idea
This is why “standardize before PCA” is advice and not a theorem: standardizing is a crude way of
doing by hand what Ψ does by fitting.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 65 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 67

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
And what it costs
What the noise model costs: W and WR fit the data identically for any rotation R, so the factors are not identified
two completely different answers to
"what does factor 1 mean?"
0.0 largest entry|WW1.1e-16⊤−(WR)(WR)--- the SAME⊤| model1e 15 1.0
x1
1 0.2 0.8
factor 0.4 x2 0.6
on x3
0.6 0.4
x4 loading
0.8 factor 1, as fitted 0.2
factor 1, after a 37 ◦ rotation x5
0.0
x1 x2 x3 x4 x5 x1 x2 x3 x4 x5
For any orthogonal R, (WR)(WR)T = WRRTW T = WW T. So W and WR fit the data identically —
the largest disagreement between the two covariances is at the level of floating-point noise.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 66 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 68

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The factors are not identified
What the noise model costs: W and WR fit the data identically for any rotation R, so the factors are not identified
PCA’s components are ordered (by variance) and
0.0 two "whatcompletelydoes differentfactor 1 mean?"answers to largest entry|WW1.1e-16⊤−(WR)(WR)--- the SAME⊤| model1e 15 1.0
x1
0.8 unique up to a sign. “The first principal 1 0.2
0.6 0.4
x3 component” means something. factoron x2
0.4
x4
0.2 Factor analysis’s loadings are unique only up to an loading 0.60.8 factor 1, as fitted
x5 factor 1, after a 37 ◦ rotation
orthogonal rotation. “The first factor” does not x1 x2 x3 x4 x5 x1 x2 x3 x4 x5 0.0
mean something.
That is why factor analysis comes with rotation
methods — varimax and friends — which pick,
among all the equally-good answers, one that is
easy to name.
Watch out
A rotation is chosen for interpretability, not for fit — every rotation fits exactly as well. If someone
names a factor, they chose that name from a continuum.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 67 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 69

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
PCA or factor analysis?
PCA factor analysis
a model of the data? no, a rotation yes, generative
noise none (or isotropic) one variance per feature, Ψ
fitted by eigen-decomposition / SVD EM (session 10)
components ordered, unique up to sign unique only up to rotation
rescaling one feature changes the answer largely absorbed
used for compression, denoising, plotting naming latent constructs
Intuition
Be honest about the field: in most machine-learning pipelines you will reach for PCA. Factor analysis
is the tool of psychometrics and the social sciences, where the meaning of the latent variable is the
deliverable.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 68 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 70

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
When negative numbers make no sense
Pixel intensities, word counts, spectra, play counts: Every entry non-negative, and the product is exact --- multiply the first row of W by the first column of H: 2 · 1 + 0 · 0 = 2
all non-negative, and a negative amount of them 2W (4×2)0 H (2×4) 2 X (4×4),4 rank0 2 2
means nothing. 1 1 1 2 0 1 1 3 3 2
0 3 × 0 1 3 1 = 0 3 9 3 PCA will happily hand you components with
negative entries, and reconstruct a face by 1 2 1 4 6 3
subtracting one pattern from another.
NMF refuses: X ≈WH with X, W , H ≥0.
Nothing can cancel — so every sample has to be
built by adding up parts.
Key idea
The constraint is the whole method. NMF is not “PCA with a positivity option” — the non-negativity
is what forces the basis to become parts.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 69 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 71

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
NMF by hand
Every entry non-negative, and the product is exact --- multiply the first row of W by the first column of H: 2 · 1 + 0 · 0 = 2
W (4×2) X (4×4), rank 2
2 0 H (2×4) 2 4 0 2
1 1 1 2 0 1 1 3 3 2
0 3 × 0 1 3 1 = 0 3 9 3
1 2 1 4 6 3
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 70 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 72

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two things that example shows
It is exact here. X really is WH — a 4 × 4 matrix of rank 2 written as a 4 × 2 times a 2 × 4, with
every entry a non-negative integer. Multiply the first row of W by the first column of H:
2 · 1 + 0 · 0 = 2.
It is not unique. For any positive diagonal D,
(WD)(D−1H) = WH = X
so you can scale a part up and its weights down and nothing changes. Unlike PCA, the factors are
not ordered and there is no “first” component.
Watch out
And the objective minW ,H≥0∥X −WH∥2F is not convex. Two runs from different random starts give
genuinely different factorizations — on the digits, the worst-matched pair of components has cosine
similarity 0.594.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 71 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 73

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What the two bases look like
PCA components mix +/ − pixels (holistic); NMF learns additive parts
±)
PCA(holistic,
≥0)
NMF (parts,
Components learned from images that are non-negative sums of localized blobs. PCA mixes +/−pixels
that cancel (top, holistic); NMF’s non-negativity forces each component to be one interpretable part
(bottom).
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 72 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 74

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Parts versus holistic, measured
PCA subtracts, so its basis is holistic. NMF cannot subtract, so it has to build the digit out of parts.
PCA: 46.7 percent of entries are negative and NMF's are LOCALIZED
the first five basis images
(blue = negative, red = positive) 102
35
PCA count 101 a 30 31.2
of64)
100 (of 25
0.2 0.0 0.2 0.4
entry value percentmass 20
90
NMF: 0.0 percent of entries are negative for image's 15 13.2
no negatives anywhere 102 pixelsbasis 10
NMF count 101 5
100 0
0 50 100 150 200 250 300 350 PCA NMF
entry value
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 73 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 75

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why the sign is the whole story
on the digits, k = 10 PCA NMF
share of basis entries that are negative 46.7% 0%
pixels carrying 90% of a basis image’s mass (of 64) 31.2 13.2
PCA spreads each component over most of the image, because it can rely on later components
subtracting the excess back off.
NMF cannot subtract, so a component that is wrong anywhere is simply wrong. The cheapest way
to be right is to be local.
Intuition
Interpretability is not a bonus feature here — it is a consequence of the constraint. Take the constraint
away and you get PCA’s holistic basis back.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 74 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 76

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where NMF is the right tool
data a “part” is W row i says
word counts per document a topic how much of each topic is in document i
an audio spectrogram a note or timbre which notes sound at frame i
images of faces a facial feature how much of each feature face i has
user × item ratings a taste how much user i has that taste
All four share one property: the data is non-negative and “how much of each” is the natural
question.
A practical note: for counts, the Kullback–Leibler loss usually fits better than the squared error,
because counts are not Gaussian.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 75 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 77

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
PCA is linear — and that is its limit
PCA can only rotate and project: it fits a flat subspace (a line, plane, hyperplane)
through the cloud.
When the data lies on a curved surface, a flat projection folds it over — distant
points on the surface land on top of each other.
The manifold hypothesis: high-dimensional data often lives near a
low-dimensional curved manifold; to unfold it we need a nonlinear method.
Intuition
Think of a swiss roll: a 2-D sheet rolled up in 3-D. PCA squashes the roll and scrambles which points
were neighbors; unrolling it needs a method that follows the surface, not a straight cut through it.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 76 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 78

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
PCA squashes, manifold learning unrolls
Linear PCA cannot unfold a curved manifold; Isomap can
PCA (linear): the roll Isomap (manifold):
3-D swiss roll collapses onto itself the sheet unrolled
The swiss roll (left): PCA’s flat projection (center) overlaps the colors; Isomap (right) recovers the true
2-D layout.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 77 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 79

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The nonlinear toolbox
Isomap: build a neighbor graph, then embed using geodesic (along-the-surface) distances instead
of straight-line ones.
LLE (locally linear embedding): reconstruct each point from its neighbors and preserve those local
weights in low dimensions.
t-SNE: match neighborhood probabilities; superb at revealing clusters for the eye.
UMAP: similar goals to t-SNE, usually faster and better at keeping some global structure.
Watch out
t-SNE and UMAP are for visualization only. They are stochastic and hyperparameter-sensitive;
cluster sizes and between-cluster distances are not meaningful. Do not use their coordinates as
general-purpose features.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 78 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 80

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
t-SNE: the plan
Goal: a 2-D map in which points that were neighbors in high-D stay neighbors.
Step 1. Turn high-D distances into a probability distribution P over pairs: pij = “how likely is it
that i and j are neighbors?”
Step 2. Build the same kind of distribution Q for the map points y1, . . . , yn ∈R2, using a
heavy-tailed kernel.
Step 3. Move the yi by gradient descent until Q matches P, measured by the KL divergence.
Intuition
Distances are never matched directly — only neighborhood probabilities are. Lineage: SNE (Hinton
& Roweis, 2002) →t-SNE (van der Maaten & Hinton, JMLR 2008), which fixes SNE’s crowding
and optimization problems.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 79 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 81

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What we are aiming for
Visualizing 64-D digits in 2-D: PCA vs t-SNE
PCA (linear): classes overlap t-SNE: same-digit islands
9
8
7
6
5
digit
4
3
2
1
0
The 64-D digits in 2-D: linear PCA (left) overlaps the classes; t-SNE (right) pulls each digit into its own
island. Great for seeing structure — but read the neighborhoods, not the gaps.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 80 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 82

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Step 1: neighbors as probabilities
Center a Gaussian on xi and ask how likely xj is as its neighbor:
exp −∥xi −xj∥2/2σ2i
pj|i = , pi|i = 0.
Pk̸=i exp −∥xi −xk∥2/2σ2i
Every point gets its own bandwidth σi: small in dense regions, large in sparse ones.
Each Pi = {pj|i}j is a probability distribution over the other points, Pj pj|i = 1.
Intuition
Read pj|i as: if i picked one neighbor at random, weighting by a Gaussian around itself, how often
would it pick j?
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 81 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 83

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Perplexity: how each σi is chosen
You set one number, the perplexity; every σi is then solved for.
Perp(Pi) = 2H(Pi ), H(Pi) = − X pj|i log2 pj|i (Shannon entropy, in bits).
j
A fair k-sided die has entropy log2 k and perplexity k: perplexity is a smooth measure of the
effective number of neighbors.
Perp grows monotonically with σi, so each σi is found by binary search until it hits the target.
Typical values 5–50 (original paper); sklearn default 30.
Watch out
Because σi adapts to local density, t-SNE equalizes dense and sparse regions — the reason cluster
sizes in the map carry no information.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 82 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 84

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Perplexity is the knob that matters
Same three clusters, four perplexities: the knob that matters most
perplexity = 2 perplexity = 5 perplexity = 30 perplexity = 100
too local: sub-clumps appear local; clusters still loose balanced (sklearn default) very global: detail compressed
Too small and the map fragments into sub-clumps; too large and local detail is compressed away. There
is no single “right” value — inspect several.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 83 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 85

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Step 1b: one joint distribution P
The conditionals are not symmetric (pj|i ̸= pi|j, because σi ̸= σj). t-SNE folds them into a single joint
distribution over pairs:
pj|i + pi|j
pij = , pii = 0, X pij = 1.
2n
i̸=j
Dividing by 2n guarantees Pj pij > 2n1 for every point.
Why that matters: an outlier is far from everyone, so its pj|i are all tiny; without the floor it would
barely enter the cost and could be dumped anywhere in the map. The floor forces the map to
place it responsibly.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 84 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 86

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Step 2: the map kernel is a Student-t
In the map, similarity uses a Student-t with one degree of freedom:
−1
1 + ∥yi −yj∥2
qij = −1 , qii = 0.
Pk̸=l 1 + ∥yk −yl∥2
A Student-t with ν degrees of freedom has density f (z) ∝ 1 + z2/ν −(ν+1)/2. At ν = 1 this is the
Cauchy distribution (1 + z2)−1; as ν →∞it tends to a Gaussian. (sklearn uses
ν = max(k −1, 1) for a k-D map, so ν = 1 exactly in the usual 2-D case.)
−1
Its tail decays polynomially, 1 + d2 ∼d−2, rather than exponentially — distant pairs keep a
non-negligible similarity.
Bonus: no exponential to evaluate, and it yields a clean gradient.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 85 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 87

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why heavy tails: the crowding problem
Volume grows like r m, so a high-D neighborhood has vastly more room than a 2-D one — in 10-D
you can place 11 mutually equidistant points; in 2-D you cannot.
With a Gaussian kernel in the map, all those moderately-distant points get crushed toward the
center: the crowding problem. The heavy tail buys back the missing room.
The Student-t kernel gives the map room: the cure for crowding
Two kernels: light vs heavy tail Same similarity, much larger map distance
1.0 high-D: Gaussian e−d 2/2σ2 1.0
map: Student-t, 1 dof (1 + d 2)−1
0.8
0.8
0.6
0.6
heavy tail: 0.4 (unnormalized) still non-negligible (unnormalized)
0.4
one similarity level = 0.02 similarity 0.2 Gaussian is 0.2 2.5× more room in the map already ≈0 similarity 0.0
Gaussian Student-t
0.0 0.2 d = 2.8 d = 7.0
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
distance d distance d
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 86 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 88

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Step 3: match Q to P with the KL divergence
C = KL(P ∥Q) = X pij log pij .
qij
i̸=j
The KL divergence is asymmetric, and that asymmetry is the character of t-SNE:
pij large but qij small ⇒a huge penalty: splitting up true neighbors is heavily punished.
pij small but qij large ⇒a small penalty: placing distant pairs close together is nearly free.
Key idea
So t-SNE spends its effort on local structure and is free to distort large distances — the formal reason
the gaps between clusters must not be read as distances.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 87 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 89

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The gradient is a force layout
∂C = 4 X pij −qij yi −yj 1 + ∥yi −yj∥2 −1.
∂yi
j
pij > qij: the pair is closer in the data than in the map ⇒attraction. pij < qij ⇒repulsion.
−1
The factor 1 + ∥yi −yj∥2 damps the force between far-apart points, so distant clusters cannot
tear the map apart.
Intuition
Springs: every pair pulls or pushes with strength pij −qij, and the map settles where all the forces
balance.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 88 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 90

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Optimizing it in practice
Initialization: the paper draws yi ∼N(0, 10−4I); modern practice (and sklearn’s default) uses a PCA
initialization — reproducible and better globally.
Gradient descent with momentum: α = 0.5 for the first 250 steps then 0.8; learning rate η starts at 100
and adapts; T = 1000 iterations.
Early exaggeration: multiply every pij by a factor early on (paper: ×4 for 50 steps; sklearn: ×12 for 250).
Q cannot match such large pij, so clusters contract and wide empty space opens up — clusters can then
slide past one another and find a good global arrangement.
Cost: exact t-SNE is O(n2) in time and memory. Barnes–Hut t-SNE (2014) summarizes distant groups
with a quad-tree (accuracy knob θ) for O(n log n); FIt-SNE uses FFT-based interpolation for O(n).
Watch out
Standard practice: first reduce to ∼50 dimensions with PCA — it suppresses noise and makes the
neighbor search far cheaper.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 89 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 91

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Reading a t-SNE map — and how to be fooled
Reading a t-SNE map: clumps can be noise, and cluster size means nothing
(a) Pure noise, perplexity = 2 (b) Same noise, perplexity = 50 (c) Spreads differ 10 ×
fake clusters! structureless, as it should be but the blobs look alike
Cluster sizes mean nothing (density is equalized), and distances between clusters are unreliable.
A small perplexity can manufacture clusters out of pure noise — always try several perplexities and seeds,
and run enough iterations.
Visualization only: no out-of-sample transform, and the coordinates are not features.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 90 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 92

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The gaps go too — not just the sizes
Not just cluster SIZE --- the GAPS go too. A 5 × difference in separation is drawn as no difference at all.
both ratios come back as 1
in the data
24.8× in the map
the DATA: radii (1, 5, 25), gaps 60 and 302 the t-SNE map: three equal blobs,
equally spaced 50 scale) 101
x2 0 (log 5.0×
50 ratio
0 100 200 300 400
x1 (the axis the clusters sit on) 100 1.00× 0.99×
size ratio gap ratio
C3/C1 d23/d12
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 91 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 93

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What that experiment measured
Three clusters in 20 dimensions, built with radii in the ratio 1 : 5 : 25 and the two gaps between them in
the ratio 1 : 5.
in the data in the t-SNE map
size ratio, biggest cluster over smallest 24.8× 1.00×
gap ratio, far pair over near pair 5.0× 0.99×
The size result is the one usually quoted. The gap result is the one that actually misleads people:
two groups drawn far apart are not more different than two groups drawn close together.
Both ratios come back as 1, with PCA initialization and with random initialization alike.
Watch out
So there are exactly two things a t-SNE or UMAP picture is entitled to tell you: which points are
near which, and how many groups there seem to be — and the second only if the answer survives
changing the knobs.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 92 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 94

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
UMAP: a neighbor graph, laid out
is what makes a dense region and a sparse one comparable.
Also nonlinear and neighbor-based, but it works in UMAP is two steps. The σi search that
so everyσipointADAPTS:still sums0.06 toto 1.63log2 k = 3.907 k-means stepon the2: laylayout scoresgraph ARIout 1.0000
two separable steps. edgestepopacity1: theisfuzzythe membershipk-NN graph pij 1.61.4
point 1.2
1.0 Step 1 — the graph. Build a weighted thatfor
0.6 k-nearest-neighbor graph and turn the distances into chosenσi 0.8
memberships in [0, 1]. 0.40.20.0 1
Step 2 — the layout. Place points in 2-D so that a (howdistancesparse2 toit theis3 around15th 4neighborthis point)5
cross-entropy between the graph’s memberships and
the layout’s is small: edges attract, randomly
sampled non-edges repel.
Two steps, two knobs — and each knob belongs to
exactly one step.
Intuition
t-SNE turns distances into probabilities and matches them. UMAP turns them into a graph and draws
it. The pictures end up looking similar because both are, in the end, attraction along neighbors against
repulsion everywhere else.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 93 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 95

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Step 1: the fuzzy neighbor graph
For each point i, with ρi the distance to its nearest neighbor:
pj|i = exp −max(0, dij −ρi)
σi
Subtracting ρi guarantees every point has at least one neighbor with membership exactly 1 — so
the graph is always connected.
σi is found by binary search, so that Pj pj|i = log2 k for every point.
Then symmetrize: pij = pj|i + pi|j −pj|i pi|j.
You have seen that binary search before: it is exactly the trick t-SNE uses to hit a target
perplexity. Same problem, same solution — adapt a bandwidth per point so dense and sparse
regions are treated alike.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 94 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 96

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The graph, actually built
UMAP is two steps. The σi search is what makes a dense region and a sparse one comparable.
step 2: lay that graph out σi ADAPTS: 0.06 to 1.63
k-means on the layout scores ARI 1.0000 so every point still sums to log2 k = 3.907
1.6
step 1: the fuzzy k-NN graph
edge opacity is the membership pij 1.4
point 1.2
that 1.0
for 0.8
0.6 chosen
σi 0.4
0.2
0.0
1 2 3 4 5
distance to the 15th neighbor
(how sparse it is around this point)
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 95 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 97

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Step 2, and the two knobs
Two knobs, two different jobs: n_neighbors decides what the graph says; min_dist decides only how tightly
the layout is allowed to pack. The same graph is used for all three pictures in the bottom row.
n_neighbors = 5 n_neighbors = 15 n_neighbors = 50
mean graph degree 6.8 mean graph degree 20.5 mean graph degree 63.2
GRAPH)
the n_neighbors
(changes
min_dist = 0 (a = 1.93, b = 0.79) min_dist = 0.25 (a = 1.12, b = 1.06) min_dist = 0.8 (a = 0.23, b = 1.68)
meanLAYOUT) within-cluster spread 0.99 mean within-cluster spread 1.31 mean within-cluster spread 1.97
min_disttheonly
hanges
n_neighbors changes the graph — at 5 the mean degree is 6.8 and pieces of it break off; at 50 it is
63.2 and the picture is global. min_dist changes only the layout: all three lower panels use the same
graph and differ only in how tightly points may pack.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 96 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 98

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which knob does what
n_neighbors min_dist
changes the graph only the layout
small values give local detail, fragments tight, dense clumps
large values give global shape loose, spread-out clouds
affects the topology? yes no
So min_dist is a presentation setting. Changing it cannot reveal or destroy structure — only
make the same structure look tighter or looser.
n_neighbors is the real one: it decides what counts as “nearby”, and therefore what the algorithm
is being asked to preserve.
Watch out
If turning a knob changes how many clusters you see, you have learned something about the knob,
not about the data.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 97 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 99

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The same warning, for UMAP
The warning, measured: a cluster's SIZE on one of these pictures tells you nothing about its spread in the data
the UMAP layout the middle blob is 3x the others
spreads [1.11, 1.13, 1.12] --- a 1.0× range in the data, and the same size in the map
1.2 in the data the DATA (first two PCs)
spreads [1.18, 3.31, 1.11] --- a 3.0× range 1.0 in the map (relative)
spread 0.80.6
0.4
0.2 within-cluster
0.0
blob 1 blob 2 blob 3
Three blobs whose spreads differ by 3.0× in the data come out 1.0× apart in the map. The σi search
that makes dense and sparse regions comparable is exactly what erases the difference.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 98 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 100

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
t-SNE or UMAP?
t-SNE UMAP
speed at scale slower usually faster
new points no out-of-sample transform has a transform
main knob perplexity n_neighbors
cluster sizes and gaps unreliable unreliable
Watch out
You will read that UMAP “preserves global structure better”. Treat that carefully: Kobak and
Linderman (2021) showed much of the measured difference comes from the initialization — t-SNE
started from PCA recovers most of it. What is not in dispute is speed, scaling, and the out-of-sample
transform.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 99 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 101

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Kernel PCA: PCA in a feature space
PCA is linear. To capture curved structure, first map each point through a feature map ϕ (as in
Lecture 07) into a higher-dimensional space and do PCA there — the principal directions become
curved back in the input space.
Kernel trick: never build ϕ. PCA in feature space needs only inner products
ϕ(xi)Tϕ(xj) = K(xi, xj) — e.g. the RBF kernel K(x, x′) = exp(−γ∥x −x′∥2).
Recipe: form the n × n kernel (Gram) matrix K, center it, and eigendecompose it — its
eigenvectors give the nonlinear components. (A dual eigenproblem on K, not the d × d
covariance.)
Watch out
Powerful for nonlinear structure, but O(n2) in memory/time, needs a chosen kernel and bandwidth
γ, and has no simple inverse map back to the input space.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 100 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 102

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Kernel PCA: unrolling the rings
Kernel PCA runs PCA in an implicit feature space (nonlinear PCA)
(a) Original: rings, not separable (b) Linear PCA: still concentric (c) Kernel PCA (RBF): unrolled
2
feature PC2 kPC2
feature 1 PC1 kPC1
Two concentric rings are not linearly separable, and linear PCA (a mere rotation) leaves them concentric.
Kernel PCA with an RBF kernel unrolls them, and a straight cut now separates the classes. Note which
ring collapses: at γ = 8 the outer points are mutually far apart, so the kernel makes each one an island
and they share a coordinate — the outer ring occupies 118× less area in the map than the inner one.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 101 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 103

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Independent Component Analysis (ICA)
PCA finds uncorrelated directions of maximum variance (second-order statistics) — orthogonal
and ordered by variance.
ICA seeks statistically independent components (higher-order statistics) by maximizing
non-Gaussianity — not orthogonal, not variance-ordered.
Model x = As: observations mix independent, non-Gaussian sources s; ICA learns the unmixing
W ≈A−1 to recover them (PCA often whitens first).
PCA decorrelates; ICA finds independent (non-orthogonal) directions
PCA: orthogonal, max variance ICA: independent directions (edges)
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 102 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 104

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
ICA in action: blind source separation
ICA: blind source separation (the cocktail-party problem)
Hidden sources (sine, square, sawtooth)
Observed mixtures (what the sensors record)
ICA-recovered sources (from mixtures alone)
time
Three independent sources are linearly mixed into three channels; ICA recovers the originals from the
mixtures alone (up to order and sign) — something PCA, which only decorrelates, cannot do.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 103 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 105

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
In practice — linear vs nonlinear
Standardize first unless features already share units — PCA follows variance.
Use PCA as preprocessing: fewer, decorrelated inputs speed up and stabilise downstream models, and drop
noise.
Reach for PCA when you want a fast, linear, reversible summary and interpretable directions.
Reach for manifold methods (t-SNE, UMAP, Isomap) when you only need a 2-D picture of curved
structure — for exploration, not features.
Reach for kernel PCA for nonlinear structure you still want to project onto, and ICA when you need
independent sources (e.g. signal separation) rather than max-variance directions.
Intuition
Default to PCA; it is cheap and honest. Bring in a nonlinear method when a flat projection clearly
cannot capture the shape — and read its plots with care.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 104 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


#### Página física 106

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Summary
Key idea
Dimensionality reduction compresses the columns: features differ in value (signal vs
redundant/noise); keep a few informative combinations — the partner to clustering.
PCA finds orthogonal directions of maximum variance; equivalently it minimizes
perpendicular reconstruction error.
Eigenvectors are directions a matrix only stretches (Av = λv). Center, form C = n1 ˜X T ˜X; its
eigenvectors are the PCs, eigenvalues the variances (C = V ΛV T); project Z = ˜XVk.
The SVD ˜X = UΣV T (rotate–scale–rotate) computes PCA (λk = σ2k/n) and gives the best
low-rank fit.
Choose k from the explained-variance ratio / scree plot (e.g. 90–95% cumulative);
standardize first.
PCA is linear; curved manifolds need nonlinear methods (Isomap, LLE, t-SNE, UMAP) —
the last two for visualization only.
Beyond PCA: kernel PCA (nonlinear, via kernels), ICA (independent, non-Gaussian
sources), NMF (additive parts, X ≈WH ≥0); t-SNE/UMAP for nonlinear visualization.
Jan Hurtado (UTEC) Dimensionality Reduction September 25, 2026 105 / 107
```

**Visual:** ejes/componentes representan proyecciones; flechas muestran transformación de muchas columnas a pocas. En mapas de calor, intensidad representa magnitud según barra de color; no inferir signo cuando se muestra valor absoluto.


### 4. ML_Course_Logistic_Regression_and_Classification.pdf — 40 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 04 — Logistic Regres-
sion & Classification
From a linear score to a probability, and how to
judge a classifier
Jan Hurtado
UTEC
August 30, 2026
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Roadmap
1 From regression to classification
2 The logistic model
3 Training: cross-entropy loss
4 Watching it learn
5 Evaluating a classifier
6 Beyond two classes
7 Wrap-up
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 2 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Key terms (for newcomers)
Class / label: the category we predict. Recall: of the true positives, how many we
catch. Probability: a number in [0, 1], the model’s
confidence. F1: harmonic mean of precision and recall.
Decision threshold: cutoff (default 0.5) Confusion matrix: counts of TP, FP, FN, TN.
turning a probability into a class. ROC / AUC: performance over all thresholds
Accuracy: fraction of predictions that are (0.5 chance, 1 perfect).
correct. Class imbalance: one class much rarer, so
Precision: of those called positive, how many accuracy misleads.
truly are.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 3 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A different question
regression: ˆy ∈R Sessions 02–03 predicted a number ˆy ∈R: a price,
a temperature, a height.
Now the target is a category: spam / not-spam, classification: y ∈{0, 1}
benign / malignant, click / no-click — so
y ∈{0, 1}. 0 1
The label is only a name; we code it 0/1 and want
a probability of the “1” class, then a decision. a dial vs. a switch
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 4 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The samples: a number vs a category
regression: y is a number (y ∈ ) classification: y is a category (y ∈ 0, 1 ) continuous y-axis only two levels © ª
6
1
5
y 4 y class 0
class 1target target
3
2 0
1
1 2 3 4 5 6 1 2 3 4 5
feature x feature x
Every sample is a pair (x, y), and the features x look the same as before — only the target changes. In
regression, y is a number on a continuous axis; in classification, y is a category, coded 0/1, so the
target axis has only two levels.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 5 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why not just fit a line to the 0/1 labels?
a line is not a probability
1.50 Tempting: least-squares fit ˆy = wTx + b to
1.25 ˆy > 1: not a probability y ∈{0, 1}, then threshold at 0.5. Three things
1.00
0.50 least-squares lineprediction/ 0.75 break:
0.25 Escapes [0, 1]: the line predicts 1.3 or −0.2 —
label 0.00 not a probability.
0.25 ˆy < 0
0.50 One far point tilts it: a distant point flattens
4 2 0 2 4 the line to shrink its huge residual, dragging the x feature
one far point tilts the line threshold along.
1.2 line (no outlier)
line (with outlier)
1.0 Punishes being right: squared error keeps
0.8 threshold charging a confident-correct point
0.6 shifts one far point ((ˆy −1)2 > 0).
0.4 drags the line down
0.2 Watch out
0.0 We need an output that lives in [0, 1] and a loss
0.2 that stops caring once a point is firmly correct.
4 2 0 2 4 6 8 10
feature x
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 6 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The fix: a curve that stays in [0, 1]
a line is not a probability
1.50
1.25 ˆy > 1: not a probability Overlay the logistic sigmoid σ(z) (blue) on the
1.00 same data. It cures the two problems we could
0.75
least-squares line see:
0.50 logistic (sigmoid)probability/
label 0.250.00 Stays in (0, 1): a genuine probability — never
1.3 or −0.2.
0.25 ˆy < 0
0.50 Ignores the far point: it saturates, so a distant
4 2 0 2 4
feature x point barely moves it and the threshold holds.
one far point tilts the line
1.2 line (no outlier)
line (with outlier) The third fix — not punishing confident-correct
1.0 logistic (unmoved) points — comes from the loss, next section.
0.8
threshold
0.6 shifts
one far point
the line down Key idea 0.4 drags
0.2 That blue S-curve is the sigmoid: it squashes the
0.0 linear score z = wTx + b into a probability. The
0.2 next slide gives its formula.
4 2 0 2 4 6 8 10
feature x
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 7 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The sigmoid link
The logistic (sigmoid) function 1.0 σ(z), z = w x + b ·
gentler slope (σ(0.6z))
σ(0) = 0.5 1
σ(z) = , z = wTx + b.
1 + e−z
|x)
1
= Its whole shape is set by the term e−z
P(y 0.5
= (always > 0):
σ(z) z = w · x + b
z →−∞: e−z →+∞, so
σ → 1+∞=1 0.
0.0 z = 0: e−z = 1, so σ = 1+11 = 12.
8 6 4 2 0 2 4 6 8
z = w · x + b z →+∞: e−z →0, so σ → 1+01 = 1.
Going deeper (optional)
So it is monotonic and smooth, squashing R into (0, 1) with the decision level at z = 0. Two
facts we reuse: symmetry σ(−z) = 1 −σ(z) (class 0 mirrors class 1), and the derivative σ′(z) =
σ(z) 1 −σ(z) — largest at z = 0, vanishing in the saturated tails.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 8 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why re-express the probability?
A linear score can be any real number, z = wTx + b ∈(−∞, ∞); a probability cannot,
p ∈(0, 1). Forcing z = p pushed the line outside [0, 1] (we just saw it).
Key idea
The fix, in one idea: don’t cramp the line to fit p — stretch p onto the whole real line, so a linear
score can match it exactly. We remove p’s two boundaries one at a time:
p
the ceiling at 1, undone by forming the odds ;
1 −p
p
the floor at 0, undone by taking the log, log .
1 −p
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 9 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Odds and log-odds: three views of one score
One linear score, three views --- and the log-odds is the score z = w ⊤x + b
probability p = σ(z) odds 1 −pp = ez log-odds log 1 −pp = z 12 1.0
(0, range (0, 1) range ∞) 4 range= the(−∞,linear∞)score 10 grows without
bound 2 8
p 0.5 odds 6 0 log-odds
4 2
2
4
0.0 0
4 2 0 2 4 2 1 0 1 2 4 2 0 2 4
z z z
0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9 0.1 0.3 0.5 0.7 0.9
corresponding p = σ(z) corresponding p = σ(z) corresponding p = σ(z)
1 Probability p ∈(0, 1) — bounded both ends, so a straight line cannot equal it.
p
2 Odds ∈(0, ∞) — “how many times more likely than not”: p=0.5→1 (even), 0.75→3 (i.e.
1 −p
3:1), 0.9→9. Dividing by 1 −p removes the ceiling — but odds still stay ≥0 and lopsided.
p
3 Log-odds log ∈(−∞, ∞) — the log removes the floor and makes it symmetric: p<0.5
1 −p
maps to negative, 0.5 →0, p>0.5 to positive, and p →0/1 to ∓∞. Same range as z — so we
model this as linear.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 10 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The linear score is the log-odds
Since the log-odds already ranges over all of R, we let the linear model be the log-odds:
p
log = z = wTx + b.
1 −p
Now invert to recover the probability — exponentiate to undo the log, then solve for p, one step
at a time: p u = ez exponentiate both sides (elog = u)
1 −p
p = ez (1 −p) multiply both sides by 1 −p
p = ez −ez p distribute ez
p + ez p = ez move the p-term to the left
(1 + ez) p = ez factor out p
ez
p = divide by 1 + ez
1 + ez
1
= = σ(z) divide top & bottom by ez
1 + e−z
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 11 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The sigmoid is forced; weights are odds ratios
Two things fall out of “log-odds are linear”:
The shape is not a choice. Start from a linear log-odds and invert: the probability must
come out as σ(z). The sigmoid is forced — not an arbitrary S-curve chosen for
convenience.
Each weight is an odds ratio. Raising xj by one unit adds wj to the log-odds, so it
multiplies the odds by ewj (the odds ratio) — the same factor whatever the other features
are.
Key idea
Concretely: wj = 0.7 ⇒e0.7 ≈2.0 — one more unit of xj doubles the odds. wj = 0 leaves them
unchanged (e0 = 1); wj < 0 shrinks them.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 12 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A model that outputs a probability
The logistic-regression model is
P(y = 1 | x) = σ(wTx + b), P(y = 0 | x) = 1 −σ(wTx + b).
We predict the class by thresholding at 0.5:
 1 if σ(wTx + b) ≥0.5 ⇐⇒ wTx + b ≥0,  ˆy =
0 otherwise. 
Going deeper (optional)
The boundary wTx + b = 0 is a line (a hyperplane in higher dimensions): logistic regression
is a linear classifier even though its probability surface is curved.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 13 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
One neuron, in one picture
x1
w1
w2 z
x2 Σ σ(z) P(y=1 | x)
b
1
Weighted sum z = wTx + b, then the sigmoid. This is the single building block that, stacked and
repeated, becomes a neural network later in the course.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 14 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The decision boundary: one feature
One feature: the boundary is a point
1.0
predict 0 (P < 0.5)
P(y = 1 | x) = σ(wx + b) |x)1
class 0 = 0.5
x ∗= P(y class 1 boundary
P (wx + b = 0, = 0.5)−b/w
predict 1 (P > 0.5)
0.0
6 4 2 0 2 4 6
feature x
With one feature the probability is the S-curve P = σ(z), z = w x + b, climbing smoothly from 0
to 1.
Predict rule: class 1 where P ≥0.5, class 0 where P < 0.5 — read straight off the curve.
P = 0.5 exactly where z = w x + b = 0, i.e. at the single point x ∗= −b/w: that point is the
boundary.
Intuition
Distance from x ∗is confidence: far out the curve pins to 0 or 1; right at x ∗it is a 50/50 coin flip.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 15 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The decision boundary: slicing the probability surface
A second feature makes P a
z = w1x1 + w2x2 + b = 0 surface over the plane — an
class 0
class 1 S-ramp from 0 (blue) to 1 (red).
Slice it with the flat plane
1.0
P = σ(z) P = 0.5: the surface meets it
1.0
along a single cut.
The cut is straight because
0.5 P = 0.5 |x)1 σ(z) = 0.5 ⇔z =
0.5 = w1x1 + w2x2 + b = 0 — linear.
P(y
Drop the cut straight down: its
0.0 shadow on the floor is the
4 boundary line.
2 4 0.0
0 2 Key idea
x2 0
2 2 x1 The boundary is linear because we slice
4 4 at the one height 0.5, where the sig-
moid leaves the flat equation z = 0
behind.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 16 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The decision boundary: the top-down map
Look straight down on that surface:
1.00 the height P becomes color (blue low,
boundary w · x + b = 0 red high) — this map is the surface
4 class 0
class 1 from above.
0.75
2 The black line is the same cut
wTx + b = 0 (P = 0.5): the shadow |x)
x2 0 0.50 1= we projected down.
P(y Same rule: red side predict 1, blue
side predict 0; the color fading across
2 0.25
the line is confidence.
4 Intuition
0.00 A point, then a line, then a plane — the 4 2 0 2 4
x1 same z = 0 boundary, one dimension higher
each time. So logistic regression is a linear
classifier.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 17 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why not squared error?
10 cross-entropy (convex) Squared error on the sigmoid output squared error on σ (non-convex) 0.6
gives a non-convex surface —
8 non-convex: basin,
then a bump 0.5 σ bumps and basins trap descent.
loss (can trap descent) on
6 0.4 error Cross-entropy gives a single convex
0.3 valley.
4cross-entropy 0.2 squared Watch out
2 single convex valley Fitting a sigmoid with least squares is 0.1
the trap: gradient descent can stall far
0 0.0 from the answer. We minimize cross- 10.0 7.5 5.0 2.5 0.0 2.5 5.0 7.5 10.0
weight w (model p = σ(wx)) entropy instead.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 18 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The likelihood of one label
Likelihood answers one question: how
Each bar = probability the model gives that point's true label; probable is the data we actually saw, if this
−logL = cross-entropy model were true? It is the same number as a likelihood L = Yi (bar),
1.0 probability, but read as a function of the
weights (w, b) — the data is fixed, and we
p(x) = σ(wx + b) fitted ask which model best explains it.|x)1
= P(y = 1) = p positive: bar= 0.5
= P(y = 0) = negative: barP(y 1 −p
wrong side: short bar → big loss Intuition
For one example, that is the bar height in the
0.0 plot: the probability the model puts on its true
4 3 2 1 0 1 2 3 4
feature x label. Tall bar = the model backs the truth;
a good model makes every bar tall.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 19 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
One label: the Bernoulli formula
That bar height, written as a formula. For a
label yi ∈{0, 1}, the model’s probability of
the observed label is
Each bar = probability the model gives that point's true label;
−logL = cross-entropy (pi if yi = 1, likelihood L = Yi (bar),
1.0 P(yi | xi) =
1 −pi if yi = 0.
p(x) = σ(wx + b) fitted|x)1
= P(y = 1) = p positive: bar One line covers both cases (a switch, since= 0.5
= P(y = 0) = negative: barP(y 1 −p
wrong side:
short big loss a0 = 1): bar →
0.0 P(yi | xi) = pi yi (1 −pi) 1−yi .
4 3 2 1 0 1 2 3 4
feature x
Check: yi=1 ⇒p1i (1 −pi)0 = pi;
yi=0 ⇒p0i (1 −pi)1 = 1 −pi. This is the
Bernoulli pmf.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 20 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
From likelihood to the log-loss
If the examples are independent, the probability of the whole dataset is the product of those
per-point likelihoods (the product of the bars):
n
L(w, b) = Y pi yi (1 −pi) 1−yi.
i=1
We want the w, b that maximize L. Instead we work with the negative log-likelihood −log L
— three wins at once:
log is increasing, so L and log L share the same maximizer — nothing lost;
it turns the awkward product (many numbers < 1, which underflows) into a tidy sum;
the minus sign flips “maximize a likelihood” into “minimize a loss”.
Intuition
So the training objective is −log L. Expanding it takes just three log identities — next.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 21 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Expanding the negative log-likelihood, step by step
Apply −log to the product and simplify one identity at a time:
yi (1 −pi) 1−yi the objective −log L = −log nY pi
i=1
n
= − X log pi yi (1 −pi) 1−yi log Q = P log: product →sum
i=1
n
= − X log pi yi + log(1 −pi) 1−yi log(ab) = log a + log b
i=1
n
= − X yi log pi + (1 −yi) log(1 −pi) log ac = c log a
i=1
Each summand is a switch: yi=1 keeps −log pi; yi=0 keeps −log(1 −pi).
Key idea
That sum is the cross-entropy: maximizing likelihood = minimizing it. Averaging over the n points
gives the loss J(w, b) — next.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 22 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The cross-entropy (log) loss
Write pi = σ(wTxi + b). The per-example loss is
ℓi = − yi log pi + (1 −yi) log(1 −pi) ,
and we minimize the average over the training set:
n
J(w, b) = −1 X yi log pi + (1 −yi) log(1 −pi)
n
i=1
Going deeper (optional)
This is the negative log-likelihood of the Bernoulli model yi ∼Bernoulli(pi): minimizing J
is maximum likelihood.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 23 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 24

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What the loss rewards
7
True y = 1: pay −log p — zero true y = 1: −log(p)
loss 6 confident & wrongtrue y = 0: −log(1 −p) when p →1.
loss is large
5 True y = 0: pay −log(1 −p) —
4 zero when p →0.cross-entropy 3 Confident and wrong is punished
harshly (loss →∞).
2
confident & correct
loss ≈0 Intuitionper-example 1
The loss does not just want the right
0
0.0 0.2 0.4 0.6 0.8 1.0 side of the boundary — it wants cali-
predicted probability p = σ(z) brated confidence.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 24 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 25

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Deriving the gradient, step by step
One example: ℓ= − y log p + (1 −y) log(1 −p) , with p = σ(z) and z = wTx + b.
Differentiate along the chain x →z →p →ℓ:
∂ℓ h y i p −y ∂p = − p −1−y1−p = = σ′(z) = p(1 −p). ∂p p(1 −p), ∂z
∂ℓ p −y ∂z ∂z
The two p(1 −p) cancel — = p(1 −p) = p −y — and with ∂w = x, ∂b = 1, ∂z p(1 −p)
averaging over the data gives exactly the linear-regression form:
1 1 ∂J 1
∇wJ = X(pi −yi) xi = XT(p −y), = X(pi −yi).
n n ∂b n
i i
Key idea
The loss’s p(1 −p) cancels σ′’s — leaving the same “error × feature” gradient as linear regression
(p −y mirrors the residual r), only now p = σ(Xw + b 1). No closed form (the sigmoid is nonlinear),
so we iterate w ←w −η ∇wJ — the Session 00–02 descent.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 25 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 26

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Convex — and kept finite by regularization
Cross-entropy is convex in (w, b): its Hessian is
1
∇2wJ = X pi(1 −pi) xixTi ⪰0
n
i
— a nonnegative-weighted sum of outer products, positive semidefinite everywhere. One bowl,
one global minimum, so gradient descent cannot get stuck (unlike squared error on a sigmoid,
which is non-convex).
Watch out
Regularize, exactly as in Lecture 03. If the classes are linearly separable, unregularized weights run
off to ∞(ever-sharper sigmoids). Add the ridge penalty J(w, b) + λ∥w∥22 — the same L2 term as
ridge regression: it keeps w finite, makes J strictly convex (a unique minimum), and λ is picked by
cross-validation. Standardize features first, also as in Lecture 03.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 26 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 27

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Gradient descent learns the boundary
Batch gradient descent from a deliberately poor start. Left: the boundary swings into place as P(y = 1 | x)
reshapes. Right: the cross-entropy falls monotonically toward its convex minimum. (Animated PDF — press play
in a viewer that supports it.)
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 27 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 28

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The fitted classifier
Final decision boundary Full cross-entropy descent
decision boundary descent so far
3 0.30 class 0 current
class 1
2 0.25
1 J(w) 0.20
x2 0 0.15 log-loss 1 0.10
2 0.05
3
0.00
4 2 0 2 4 0 5 10 15 20
x1 iteration t
The static summary: the final decision boundary with its probability shading (left) and the full
cross-entropy descent curve (right) — the convex valley reached by the animation.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 28 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 29

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Accuracy: the first score
The obvious first yardstick: what fraction of
Accuracy = fraction of predictions that are correct 88 predictions are correct?
100 = 0.88
correct (88) wrong (12) correct predictions
accuracy = .
all predictions
Ranges 0 to 1; higher is better, and 0.5
is coin-flip for two balanced classes.
Screening 10,000 people for a 1%
disease, a model gets 9,680 right: 96.8%
— sounds excellent.
Intuition
One number, easy to read — the right
place to start. But it counts every mis-
take the same, and cannot say which kind.
That is the catch.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 29 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 30

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Accuracy is not enough
A single number — accuracy — hides which errors a classifier makes.
On imbalanced data it is misleading: predicting “always the majority class” can
score 95% and be useless.
We need the kinds of mistakes: false positives vs. false negatives.
Watch out
A cancer screen that never says “positive” is highly accurate and completely dangerous. Look
past accuracy.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 30 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 31

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The confusion matrix
predicted class
0 (neg) 1 (pos)
Four outcomes on the held-out test set:
100 TP / TN: correct (the diagonal).
102 19 0 (neg) 80 FP: cried wolf (predicted 1, truly 0).
TN FP
FN: missed it (predicted 0, truly 1).class 60
count Key ideaactual 40
11 108 Precision = TP+FPTP (are positives real?); 1 (pos) TP
20 recall = TP+FN (did we catch them?). FN TP
0
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 31 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 32

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The four outcomes: TP, FP, FN, TN
Screening 10,000 people (1% sick: 100 sick, 9,900 healthy)
predicted: Cleared predicted: Flagged
Positive = has the disease.
9,600 300 Healthy TP: sick and flagged.
TN (true negative) FP (false positive)
FN: sick but missed — the
healthy, correctly cleared healthy, false alarm actual: dangerous error.
FP: healthy but flagged — a false disease 20 80 alarm.
Has FN (false negative) TP (true positive) TN: healthy and cleared.
actual: sick, MISSED (dangerous) sick, correctly caught
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 32 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 33

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Precision, recall, and their balance: F1
Two questions about the positives:
F1 = P2PR+ R: high only when both are high 1.0 1.00 Precision = TP+FPTP : of those flagged,
how many are real?
0.8 Recall = TP+FNTP : of the real ones, how
0.75 many did we catch?
our model 0.8
P=0.21, R=0.80 FN) F1=0.33 (mean 0.51) They trade off: lower the threshold to+ 0.6
TP/(TP 0.50 score catch more (recall up) and false alarms grow (precision down).= 0.6 F1
recall 0.4 Key idea
0.4 0.25 F1 = 2P+RP R , the harmonic mean, sits near
0.2 the smaller of the two — no faking it
0.2 by acing one and failing the other. Our
0.0 0.00 model has R = 0.80 but P = 0.21, so
0.0 0.2 0.4 0.6 0.8 1.0 F1 = 0.33: a far soberer verdict than = TP/(TP + FP) precision
96.8%.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 33 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 34

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why accuracy lies on imbalanced data
Same data, same 99% accuracy ballpark: only recall tells them apart
100 99% 97% 'AlwaysReal modelsay healthy'
80 80%
(%) 60
score
40
21%
20
0 0% 0%
Accuracy Recall Precision
(sick caught)
Screen 10,000 people, 1% prevalence (100 sick, 9,900 healthy).
“Always say healthy”: 99% accuracy yet 0% recall — catches no one.
Real model (TN=9600, FP=300, FN=20, TP=80): 97% accuracy, 80% recall, 21% precision.
Watch out
On rare-event data, report precision / recall / F1 / AUC, not accuracy.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 34 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 35

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Sweeping the threshold: the ROC curve
1.0
The 0.5 cut is just one choice; move it
0.8 and TPR/FPR trade off.
The ROC curve plots that trade-off
rate across all thresholds. 0.6
AUC (area under it) summarizes
ranking quality: 1.0 perfect, 0.5
0.4 chance.true-positive
Going deeper (optional)
0.2 AUC = 0.926 ROC/AUC judge the scores, indepen-
dent of where you place the threshold — ROC (AUC = 0.926)
chance (AUC = 0.5) pick the operating point your application
0.0 needs.
0.0 0.2 0.4 0.6 0.8 1.0
false-positive rate
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 35 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 36

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A real classifier end-to-end: breast-cancer diagnosis
Confusion matrix (test set)
predicted
malignant benign Logistic regression on 569 cases, 30 features
accuracy 0.958
50 3 malignant
precision 0.967
TN FP
recall 0.967 actual
3 87 F1 0.967 benign
FN TP ROC-AUC 0.995
0.0 0.2 0.4 0.6 0.8 1.0
score
scikit-learn breast cancer: 569 cases, 30 features; standardized logistic regression on a stratified
75/25 split.
accuracy 0.958, precision 0.967, recall 0.967, F1 0.967, ROC-AUC 0.995; confusion
TN=50, FP=3, FN=3, TP=87 — only 3 malignant missed.
Runnable end-to-end in the companion notebook session-04-logistic-regression.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 36 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 37

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Beyond two classes: softmax
For K classes, give each its own score
zk = wTk x + bk and normalize the scores with
the softmax:
softmax: three classes, linear boundaries
6 class 0 ezk
class 1 pk = , Pk pk = 1. class 2 PKj=1 ezj 4
2
Predict ˆy = arg maxk pk; each pairwise
x2
0 boundary is still a line.
Loss: cross-entropy with a one-hot y,
2 −Pk yk log pk, and the gradient is again
4 p −y — the same residual form.
6 4 2 0 2 4 6 Key idea
x1
Binary logistic is the K=2 case: σ(z) is a two-
class softmax. One score per class, softmax,
cross-entropy — everything carries over.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 37 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 38

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Summary
Key idea
Classification predicts a category; logistic regression squashes the linear score
z = wTx + b through the sigmoid into P(y = 1 | x).
The decision boundary z = 0 is linear; the probability surface is smooth.
Train by minimizing the cross-entropy (negative log-likelihood) — convex, with the
same (pi −yi)xi gradient and the Session 00–02 descent.
Squared error on a sigmoid is non-convex — avoid it.
Evaluate with the confusion matrix (precision/recall) and the ROC/AUC, not
accuracy alone.
For > 2 classes, softmax generalizes the sigmoid — same cross-entropy and p −y
gradient.
Next: beyond linear boundaries — features, kernels, and trees.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 38 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 39

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
References
C. M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. (§4.3 logistic regression
& the logistic sigmoid; §4.3.2 iterative reweighted least squares.)
T. Hastie, R. Tibshirani, J. Friedman (2009). The Elements of Statistical Learning, 2nd ed.
Springer. (§4.4 logistic regression; §9.2 ROC & classifier assessment.)
K. P. Murphy (2022). Probabilistic Machine Learning: An Introduction. MIT Press. (Ch. 10 logistic
regression; §5.1 the cross-entropy / log loss.)
G. James, D. Witten, T. Hastie, R. Tibshirani (2013). An Introduction to Statistical Learning.
Springer. (§4.3 logistic regression; §4.4.2 confusion matrices & ROC.)
T. Fawcett (2006). An introduction to ROC analysis. Pattern Recognition Letters 27(8), 861–874.
Jan Hurtado (UTEC) Logistic Regression & Classification August 30, 2026 39 / 40
```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


#### Página física 40

**Transcripción técnica de la diapositiva (extracción de texto):**

```text

```

**Visual:** curva sigmoide mapea scores a probabilidades entre 0 y 1; frontera en p=0.5 corresponde a score 0. Colores de puntos son etiquetas de clase del ejemplo.


### 5. ML_Course_Generative_Probabilistic_Models.pdf — 39 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 05 — Generative &
Probabilistic Models
Modeling how the data is generated: Bayes,
naive Bayes, and discriminant analysis
Jan Hurtado
UTEC
September 7, 2026
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Roadmap
1 A quick probability refresher
2 From a boundary to a model of the data
3 Bayes’ rule and the optimal classifier
4 Naive Bayes
5 Gaussian discriminant analysis
6 Generative vs. discriminative
7 Wrap-up
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 2 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Key terms (for newcomers)
Big P = probability of a discrete outcome (a class); little p = a density over a continuous feature. The
next three frames unpack every term below.
Generative model: models how each class Bayes’ rule: posterior ∝prior × likelihood.
produces its features, p(x | y), plus P(y). Conditional independence: features treated
Discriminative model: models the boundary as independent given the class (the “naive”
/ P(y | x) directly (e.g. logistic regression). assumption).
Prior P(y): how common each class is before Maximum likelihood (MLE): pick parameters
seeing x. that make the observed data most probable.
Likelihood p(x | y): how probable these MAP (maximum a posteriori): pick the class
features are under each class. (or parameters) with the largest posterior —
MLE weighted by a prior. Posterior P(y | x): updated class probability
after seeing x. Covariance: how features vary together; its
shape is an ellipse.
Decision boundary: where two classes’
posteriors are equal.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 3 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Probabilities vs. densities: two ways to say “how likely”
A probability is a number in [0, 1] for a
discrete outcome (a class). List one per
outcome; they sum to 1. Write it big:
P(y=1) = 0.4.
A continuous feature cannot work that way
continuous feature 0.8 discrete outcome → use P 0.5 → use p — any exact value has essentially zero
total area under the curve = 1
0.7
0.6 0.6 bars sum to 1 0.4 a single exact x has probability 0 P(a x b) = shaded area chance. Use a density p(x): a curve whose
0.5 area over a stretch is the probability of 0.4 0.4 p(x) 0.3probability 0.3 density 0.2 landing there.
0.2 0.1
0.1 Total area under the whole curve = 1; the
0.0 0.0
y = 0 y = 1 4 3 2 1 0 1 2 3 4 height is relative plausibility (it can even
x
exceed 1).
Key idea
That is why we write big P(y) for classes and
little p(x) for feature densities — the same
idea, two notations, on purpose.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 4 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Conditional probability, by counting
P(A | B) reads “probability of A given B”: shrink the world down to the cases where B holds,
then count the fraction that also have A. The bar “|” means restrict, then count. Terms: spam
= a junk email; ham = a normal, wanted email (the opposite of spam); free = whether the
word “free” appears in the message — one feature we can count.
Example
100 emails: 40 spam, 60 ham. Of the 40 spam, 32 contain the word free ⇒p(free | spam) = 3240 = 0.8.
Of the 60 ham, only 6 do ⇒p(free | ham) = 606 = 0.1.
Flip the bar and you ask a different question: P(spam | free) = of the emails containing the word
free, what fraction are spam? In general P(A | B) ̸= P(B | A) — direction matters.
Key idea
A likelihood is “given the class”, p(x | y); a posterior is “given the features”, P(y | x). Bayes’ rule
(next section) turns one into the other.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 5 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Fitting a model = counting and averaging (what “MLE” means)
Maximum likelihood (MLE) = pick the parameters that make the data you actually saw as
probable as possible. It sounds heavy, but for every model in this lecture it collapses to counting
fractions and averaging numbers — no calculus:
Nk
a prior πk = P(y=k) = — e.g. 40 spam of 100 ⇒πspam = 0.40;
N
#{class-y docs with w}
a word probability p(w | y) = — a fraction again (0.8 above);
#{class-y docs}
a mean µ = the average of the values; a variance σ2 = the average squared distance from that
mean (the spread).
Intuition
Whenever a later slide says “MLE”, read it as: count the fractions, average the numbers.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 6 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two ways to separate classes
Session 04 was discriminative: logistic regression fit P(y | x) directly, learning only
where to cut.
A generative model instead asks: how does each class produce its data? Learn the
likelihood p(x | y) and the prior P(y), then invert with Bayes’ rule — flip “features
given class” into “class given features” — to get P(y | x).
Knowing what each class looks like buys you more than a boundary: you can
generate synthetic samples, fill in missing features, flag outliers, and add a new
class cheaply.
Intuition
Discriminative learning finds where to draw the line; generative learning models what each
class looks like — the line then falls out by comparison.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 7 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Boundary vs. class-densities
Discriminative: learn the boundary Generative: model each class
model P(y | x) model P(x | y) and P(y)
3 3
2 2
1 1
x2 0 x2 0
1 1
2 2
3 3
4 3 2 1 0 1 2 3 4 4 3 2 1 0 1 2 3 4
x1 x1
Left learns a fence between the classes (a boundary in feature space); right learns a portrait of each
class — a probability density p(x | y) sitting over it — and lets the fence follow. The same data, two
different things to learn.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 8 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The generative recipe
1 Estimate the prior P(y=k) = πk — how common each class is (just count:
πk = Nk/N).
2 Model each class’ likelihood p(x | y=k) — a density per class (Gaussian,
products of word probabilities, . . . ).
3 Invert with Bayes to get the posterior P(y=k | x) and predict its arg max.
Key idea
Generative classifiers are built, not drawn: assemble prior × likelihood for every class, and
the decision boundary emerges automatically.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 9 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bayes’ rule
The chance of a class and its features, p(y=k, x), can be built two ways: pick the class then
generate its features (πk p(x | y=k)), or observe the features then ask which class
(p(x) P(y=k | x)). Setting them equal and solving for the posterior gives Bayes’ rule:
πk p(x | y=k) P(y=k | x) = .
Pj πj p(x | y=j)
| p(x){z }
The denominator p(x) = Pj πj p(x | y=j) is the total prior×likelihood over all classes: it makes
posteriors sum to 1 but is the same for every class, so it never changes the winner. To decide,
drop it:
ˆy = arg max πk p(x | y=k).
k
Going deeper (optional)
This rule minimizes the error probability among all classifiers. Where the two prior-weighted class
“portraits” overlap, even the best rule must guess — the irreducible Bayes error.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 10 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bayes’ rule, made concrete: base rates
A test that catches 80% of the sick just
fired positive. Is the patient sick? Restrict
the world to the positives
Bayes with natural frequencies: a positive test on a rare disease
(80 + 300 = 380), then count the truly sick
10,000 people (80) — conditional probability by counting:
1%
(prior) 99% P(sick | +) = 80+30080 ≈21%.
Sick Healthy 100 9,900 The test gives the likelihood
80% 3%
catch 20% false alarm 97% p(+ | sick) = 0.80; the patient wants the
posterior P(sick | +) = 0.21. Bayes flipped
Test 80 + Test20 − Test300+ Test9,600− the bar.
Intuition
A positive test: 80 + 300 = 380 people, but only 80 truly sick The disease is rare, so most positives
P(sick | + ) = 80/380 ≈ 21% are false alarms. This 21% is exactly
the precision from Session 04 — a live
case of P(A | B) ̸= P(B | A).
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 11 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Base rates as Bayes’ rule: the calculation
Bayes’ rule (reminder, so you needn’t flip back):
πk p(x | y=k)
P(y=k | x) = k ∈{sick, healthy}, x = a positive test .
Pj πj p(x | y=j)
Name the pieces:
P(sick) = 0.01, p(+ | sick) = 0.80, p(+ | healthy) = 0.03.
| prior{z π } likelihood| {z(sensitivity)} | false-alarm{z rate }
Specialize to two classes and plug in (denominator = evidence p(+)):
π p(+ | sick)
P(sick | +) =
π p(+ | sick) + (1 −π) p(+ | healthy)
0.01 × 0.80 0.008
= = ≈0.21.
0.01 × 0.80 + 0.99 × 0.03 0.0377
Key idea
Multiply each product by the 10,000 people and the tree reappears: 0.008 →80 (sick & positive),
0.0297 →297 ≈300 (false alarms), total 377 ≈380. Probabilities and counts tell the same story.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 12 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Prior × likelihood →posterior
Bayes' rule: posterior ∝ prior × likelihood
y = 0) 0.20 π0 p(x |
π1 p(x | y = 1)
likelihood 0.150.10 overlap = error
×
prior 0.05
0.00
Posterior: the same S-curve logistic regression learns
1.0
0.8
|x) 1 0.6
=
P(y 0.4 decision
0.2 boundary
0.0
4 2 0 2 4 6
x
Top: each class’s prior-weighted “portrait” πk p(x | y=k); where orange overtakes blue is the boundary.
Bottom: the posterior P(y=1 | x) climbs through 0.5 at that crossing. This S-curve is not assumed —
it falls out of two Gaussians, and it is the very sigmoid logistic regression fit directly in Session 04.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 13 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The decision rule in one line
Key idea
Pick the class with the largest prior × likelihood:
ˆy = arg max πk p(x | y=k).
k
(arg maxk = the class k whose score πk p(x | y=k) is largest.) The shared denominator p(x)
can change the numbers but never the winner, so we ignore it when deciding. Everything in
this lecture is a different choice of how to model the likelihood p(x | y=k) — the rule above
never changes.
Intuition
This connects straight back to the prior / likelihood / posterior refresher in Session 00: seeing
x updates our belief about the class from the prior πk to the posterior.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 14 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Maximum a posteriori (MAP): the name for our rule
Predicting the class with the largest posterior has a name — the maximum a posteriori
(MAP) estimate (“a posteriori” = after seeing the data):
ˆyMAP = arg max P(y=k | x) = arg max πk p(x | y=k) .
k k |{z}prior | likelihood{z }
Compare pure maximum likelihood (MLE), which ignores the prior:
ˆyMLE = arg maxk p(x | y=k). So MAP = MLE + prior; when the classes are equally common
(all πk equal), the two rules coincide.
Going deeper (optional)
The same idea also fits parameters, not just picks a class: θMAP = arg maxθ p(D | θ) p(θ) —
maximum likelihood plus a prior p(θ) that pulls the estimate toward sensible values (a regularizer).
The Laplace smoothing a few frames on is exactly this: the “+α” is a prior of α imagined counts.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 15 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The curse of the full joint
To use Bayes we need p(x | y=k) — a density over the whole feature vector
x = (x1, . . . , xd).
In high dimension this joint is hopeless to estimate. With just 20 yes/no word
features, the full joint needs 220 ≈1,000,000 cell probabilities per class; conditional
independence needs only 20 (one per word). The parameters explode.
Naive Bayes cuts the knot with a bold assumption — conditional independence
of the features given the class:
d
p(x | y=k) = Y p(xj | y=k).
j=1
Intuition
“Naive” because it pretends the features carry no joint information once you know the class
— estimate each feature’s distribution on its own.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 16 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Deciding in log-space
Multiplying many small probabilities underflows to 0 on a computer — e.g. 30 factors
near 0.1 give a product ≈10−30. Take logs: products become sums, and since log is
increasing the top-scoring class is unchanged (logs move the numbers, not the winner):
d
h ˆy = arg max log πk + X log p(xj | y=k) i .
k j=1
The rule is the same for every naive Bayes; only the per-feature model p(xj | y=k)
changes:
Continuous xj →Gaussian NB: xj | y=k ∼N(µkj, σ2kj).
Text (words) →Bernoulli / Multinomial NB: presence or counts of each word.
We take them in order — continuous (Gaussian) next, then text. Either way the fit is
MLE = counts and moments (πk = Nk/N; class mean/variance; word frequency).
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 17 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The normal distribution in 1-D
Continuous feature →Gaussian NB models each xj as a bell N(µ, σ2) — so first, what is a bell?
The normal (Gaussian) density p(x) = σ 12π e−(x −µ)2/2σ2
one bell: center µ, spread σ p the two knobs
0.5 mean
(center) µ 0.5 µ moves the center; σ sets the width N(0, 1)
0.4 0.4 N(1.2, 1.82)
bigger σ:
p(x) 0.3 0.3 wider & shorter
density 0.2 ±1σ: 68% 0.2
0.1
±2σ: 95% 0.1
0.0 width = 2σ
0.0
µ−2σ µ−σ µ µ+σ µ+2σ 4 3 2 1 x0 1 2 3 4
The normal (Gaussian) is the bell-shaped density 68 / 95 / 99.7 rule: ≈that % of samples fall
for one continuous feature, x ∼N(µ, σ2): two within ±1, ±2, ±3 σ of µ. The scary exponent is
knobs only — the center µ and the spread σ just the z-score squared, (x −µ)/σ 2 — “how
(variance σ2). many σ from the center”.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 18 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Gaussian naive Bayes: axis-aligned densities
Model — each continuous feature is its own 1-D
Gaussian naive Bayes: axis-aligned class densities Gaussian:
class 0
4 class 1 p(xj | y=k) = N(xj; µkj, σ2kj)
= 1 e−(xj −µkj )2/(2σ2kj ).
2 p2πσ2kj
x2 0 Classify — drop it into the naive-Bayes log rule and pick
the top score:
2 h i ˆy = arg max log πk + Pdj=1 log N(xj; µkj, σ2kj) .
k
4
3 2 1 0 1 2 3
x1 Key idea
A product of per-feature bells ⇒axis-aligned Estimate (MLE) = counts & moments: prior πk = Nk/N;
ellipses (no tilt); the boundary still curves. and for each class k and feature j, the mean µkj and vari-
GDA (next) can tilt them. ance σ2kj of xj over class-k points. Total: K priors + 2Kd
numbers — nothing else.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 19 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Naive Bayes for text: Bernoulli & multinomial
Text has no natural x1, x2 axes — first turn a document into features. Bag of words: fix a
vocabulary; each word becomes a feature.
Bernoulli NB — each feature is 0/1 = “is this word present?”; then p(xj | y) is simply how often
word j shows up in class y (one fraction per word).
Multinomial NB — features are word counts (how many times each word occurs); better when
repetition matters.
Either way, naive Bayes multiplies one simple per-word factor per class, so there are only about
(vocabulary size × classes) numbers to estimate — each just a count.
Intuition
A spam filter doesn’t parse grammar. It asks, word by word, “does this appear more in spam or in
ham?” and multiplies the votes together.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 20 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Worked example: is this email spam?
Model (Bernoulli NB) — words independent given the
class, so for the message “free money now”:
p(words | y=k) = Y p(wj | y=k),
j
Per-word likelihoods (estimated from counts) Message: "free money now" with p(wj | y=k) = fraction of class-k emails
free 1.0 0.99 containing wj. Classify — normalize prior × likelihood
money 0.8 (Bayes), predict the larger:
now probability 0.6
meeting πspam Qj p(wj | spam)
0.4 project posterior P(spam | words) =
0.2 Pc∈{spam,ham} πc Qj p(wj | c).
P(word | spam) report
P(word | ham) 0.01
0.0 0.2 0.4 0.6 0.8 1.0 0.0 spam ham
likelihood of the word
Key idea
Estimate (MLE): πk = Nk/N (share of emails in class
k); each p(wj | y=k) = (smoothed) fraction of class-
k emails with wj. For a vocabulary of V words: K
priors + KV word-probabilities. Numbers next ⇒≈0.99
spam.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 21 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Spam by hand: the actual multiplication
Priors πspam = 0.40, πham = 0.60. For the present words, the likelihoods (from counts, as in the figure)
are
free money now
spam 0.80 0.70 0.55
ham 0.10 0.08 0.30
Each class’s score = prior × its word-likelihoods:
spam score = 0.40 × 0.80 × 0.70 × 0.55 = 0.1232,
ham score = 0.60 × 0.10 × 0.08 × 0.30 = 0.00144.
Normalize (the Bayes denominator is just the two scores added):
0.1232
P(spam | words) = ≈0.99.
0.1232 + 0.00144
Key idea
That is ˆy = arg maxk πk p(x | y=k) with real numbers plugged in — verdict: spam (odds ≈86:1).
Nothing here is beyond a calculator.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 22 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Smoothing, and a caution
If lottery never appeared in any ham email, its count is 0, so p(lottery | ham) = 0 —
and that single 0 zeroes the whole ham product, however ham-like the other words are.
Laplace (add-α) smoothing fixes it:
count(w, k) + α
p(w | y=k) = ,
total(k) + αV
where V is the vocabulary size (α = 1 is common). This is a MAP estimate: the +α
acts like a prior of α imagined counts per word, so no probability is ever exactly 0.
Watch out
Conditional independence is almost always false (“free” and “money” co-occur). Yet naive
Bayes often classifies well: the arg max can be right even when the probability is badly
miscalibrated — typically over-confident. Trust the label, not the 0.99.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 23 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 24

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Beyond naive Bayes: correlated features need covariance
The covariance Σ shapes the Gaussian: circle → axis-aligned → tilted ellipse
spherical: equal spread, no tilt diagonal (naive Bayes): axis-aligned full: correlated ⇒ tilted
x2 x2 x2
Σ = [1.0, 0.0; 0.0, 1.0] Σ = [2.4, 0.0; 0.0, 0.5] Σ = [1.5, 0.9; 0.9, 1.0]
x1 x1 x1
Diagonal σ21, σ22 = each feature’s spread; off-diagonal σ12 = how they move together. σ12=0 ⇒
axis-aligned (naive Bayes); σ12̸=0 ⇒tilted — the shape only GDA can fit.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 24 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 25

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Reading the exponent: distance in units of spread
A Gaussian scores a point by one number in its exponent — the Mahalanobis distance.
x−µ 2
1-D: it is the squared z-score, D2 = σ — “how many σ from the center”.
Many-d: replace “divide by σ” with Σ−1, which rotates onto the ellipse’s axes and divides by the
spread along each: D2(x) = (x −µ)TΣ−1(x −µ).
The full density just puts D2 in the exponent; |Σ| is the volume normalizer:
1 N(x; µ, Σ) = e−12 D2(x).
(2π)d/2 |Σ|1/2
Key idea
The density depends on x only through D2, so its contours {x : D2(x) = c} are exactly the ellipses.
“One spread” along a wide direction covers more ground than along a narrow one — so equal
Euclidean distances are not equal Mahalanobis distances (next).
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 25 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 26

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Euclidean circles vs. Mahalanobis ellipses
Mahalanobis distance: how many spreads, ellipse-aware, from the center?
Euclidean: circles ignore the cloud's shape Mahalanobis: D 2 = (x −µ) ⊤Σ−1(x −µ)
long / high-variance axis:
go farther for the same D
B A B: D = 2.1 A: D = 1.0
x2 x2
∥A −µ∥= ∥B −µ∥= 1.6 (same circle)
x1 x1
Same tilted cloud. A and B are the same Euclidean distance from the center (left: same circle), yet A
(along the fat axis) sits at Mahalanobis D=1 while B (across the thin axis) sits at D≈2.1: a step along
a high-variance direction is “ordinary”, so it costs less distance. Mahalanobis = Euclidean measured in
the ellipse’s own units.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 26 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 27

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Modeling each class as a full Gaussian: fit by counting & averaging
Give each class its own Gaussian, p(x | y=k) = N(x; µk, Σk), πk = Nk/N. MLE is again just count and
average:
µk = 1 X xi, Σk = 1 X(xi −µk)(xi −µk)T.
Nk Nk
i: yi =k i: yi =k
Two ways to use the spread:
QDA — each class keeps its own Σk (its own ellipse): N(x; µk, Σk).
LDA — all classes share one pooled Σ (identical ellipse; only the center µk moves), the
Nk-weighted average of the class covariances:
Σ = 1 X Nk Σk = X Nk Σk.
N N
k k
Key idea
Same three ingredients as naive Bayes — a center µk, a spread, a share πk — but the spread is now a full
matrix that can tilt. QDA: one per class ⇒quadratic boundary. LDA: one shared ⇒linear boundary (why:
next).
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 27 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 28

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Each class is a Gaussian; the Bayes boundary follows
Each class = a Gaussian (µk, Σk); Bayes picks the larger posterior
4
3
2
1
x2 µ1, Σ1
0 µ0, Σ0
1
2
4 2 0 2 4
x1
πk p(x | k)
Posterior by Bayes: P(y=k | x) = the boundary is the tie Pj πj p(x | j);
π0 p(x | 0) = π1 p(x | 1) ⇔P(y=1 | x) = 12. Fit one ellipse per class (µk, Σk) and the boundary follows
— everywhere else the taller weighted portrait wins.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 28 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 29

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
LDA (linear) vs. QDA (quadratic)
LDA plugs the shared Σ into
every class ⇒the xTΣ−1x term is
LDA: shared covariance QDA: per-class covariance the same for all and cancels: a
4 ⇒ linear boundary 4 ⇒ quadratic boundary straight boundary wTx + b = 0.
3 3 QDA plugs each class’s own Σk
2 2 ⇒that quadratic term survives:
1 1 a curved boundary
x2 x2
0 0 xTAx + wTx + b = 0 that can
wrap the tighter class.
1 1
2 2 Watch out
Cost: LDA fits one shared Σ; QDA fits
4 2 0 2 4 4 2 0 2 4 one per class (d(d+1)/2 numbers each
x1 x1
— 3 apiece when d=2). More knobs
= higher variance = needs more data
(Session 03’s bias–variance).
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 29 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 30

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The discriminant: a per-class Gaussian score
A discriminant δk(x) = log(prior × likelihood) (dropping the shared constant −d2 log 2π); predict the
class with the largest score. For a Gaussian,
δk(x) = log πk −12(x −µk)TΣ−1k (x −µk) −12 log |Σk| .
head| {zstart} | Mahalanobis distance{z to center } penalty| for {za loose }ellipse
Expand the middle term — the only piece quadratic in x is xTΣ−1k x (that is QDA’s curvature):
(x −µk)TΣ−1k (x −µk) = xTΣ−1k x −2 µTk Σ−1k x + µTk Σ−1k µk.
quadratic| {z in} x
Intuition
The score reads: a head start for a common class, minus how far x sits from the center in ellipse units, minus
a penalty for a big loose ellipse. Own Σk per class ⇒the quadratic term stays ⇒curved boundary (QDA).
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 30 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 31

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Sharing Σ cancels the quadratic term ⇒linear boundary
With Σk = Σ for every class, the xTΣ−1x and log |Σ| terms
are identical across classes, so they cancel. Each score is now
linear:
Sharing Σ cancels the x ⊤Σ−1x term: quadratic scores, linear difference δk(x) = wTk x + bk, wk = Σ−1µk.
Shared σ (LDA)
0 Per-class σ (QDA) The two-class boundary δ1 −δ0 = 0 is wTx + b = 0 with 15 0
10 10 40 5
5 30 w = Σ−1(µ1 −µ0),
δk(x) 10 δk(x) 20
0 −δ0δ1
30score score
15 20 −δ0δ1 b = −12 (µ1+µ0)TΣ−1(µ1−µ0) + log π0π1 . 5
10
40
10 unequal
δ0 δ0
δ1 δ1 0 20 equal⇒x2⇒ LINEcancelscurvature(one boundary) ⇒x2⇒ PARABOLAsurvivescurvature(two) 50 15
4 2 0 2 4 6 8 10 12 4 2 0 2 4 6 8 10 12 x x Going deeper (optional)
So under LDA P(y=1 | x) = σ(wTx + b) — the same
logistic form Session 04 fit directly, here derived from
Gaussian classes.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 31 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 32

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
LDA by hand: the boundary is the (weighted) midpoint
1-D (Σ →σ2), means µ0=2, µ1=6. Specialize the
general w, b (then x ∗= −b/w):
Equal shared variance ⇒ boundary at the midpoint x ∗= µ0 +2 µ1 = 4
class 0 σ2 µ0 + µ1 π1 µ1 −µ0
class 0.20 µ0 = 2 µ1 = 6 − , x∗= log . 1 w =
σ2 2 µ1 −µ0 π0
0.15likelihood
×0.10 Equal priors: the log term isprior
0.05 0 ⇒x∗= µ0+µ12 = 4 (“nearest mean wins”). overlap
= Bayes error With σ2=4: w=1, b=−4, so 0.00
P(y=1 | x) = σ(x −4) (the green curve).
1.0
w = µ1 σ2−µ0 = 1, b = −4
σ2 π1 Unequal priors: the − log term slides
µ1−µ0 π0
(shrinking its territory). 0.5 P(yσ(x =−4)1 | x) x∗toward the rarer class posterior
0.0 Intuition
2 0 2 4 6 8 10 In higher-d, w = Σ−1(µ1 −µ0) is the arrow be-
x tween the means (rescaled by the shared spread);
the boundary is its perpendicular bisector.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 32 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 33

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
LDA →QDA: relaxing the shared-covariance assumption
Blend each class’s covariance with one knob
t, from the shared pooled Σ to its own:
Same data, two covariance assumptions
4 LDA (shared Σ): straight Σk(t) = (1 −t) Σpooled + t Σk. QDA (own Σk): curved
3
2 t=0: all classes share Σpooled ⇒the
quadratic term cancels ⇒LDA
1
x2 (straight, green).
0 t=1: each class reclaims its own Σk ⇒
the quadratic term survives ⇒QDA
1
(curved, dashed).
2
Key idea
Watch: while the two ellipses stay the same
4 2 0 2 4
x1 shape the cut is straight; it bends by exactly
as much as their shapes/tilts diverge.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 33 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 34

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
LDA →QDA: the same sweep, animated
As t runs 0 →1 the two ellipses drift apart and the straight LDA cut bends into QDA’s curve.
(Animated PDF — press play in a supporting viewer; the static summary is on the previous slide.)
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 34 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 35

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two routes to the same logistic posterior
LDA (generative) and logistic regression (discriminative) reach the same form
σ(wTx + b) — so why choose one?
They differ in how they get there, and that changes their behavior with little vs.
plenty of data.
Generative learns faster; discriminative needs more data to match it
naive Bayes (generative)
0.25 logistic regression (discriminative)
0.20
error 0.15
test
0.10
0.05
101 102
training examples
Learning curves: x-axis = number of training examples, y-axis = test error. Naive Bayes drops fast and
flattens a little higher; logistic regression needs more data but ends lower — the flat level each
approaches is its asymptote. (Ng & Jordan, 2001.)
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 35 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 36

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The trade-off
Generative (NB, LDA/QDA) Discriminative (logistic reg.)
Lower sample complexity (how many Fewer assumptions — models only the
examples you need for good accuracy) — boundary.
learns fast from few examples.
Often more accurate when data is plentiful.
Can generate data, handle missing features
Wins when the generative assumptions are
and outliers.
wrong.
Adding a new class is cheap (just fit its
But: needs more data; cannot generate or
density).
impute.
But: makes strong modeling assumptions.
Intuition
Rule of thumb: little data or you need the density? go generative. Plenty of data and
you only need the label? go discriminative.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 36 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 37

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Summary
Key idea
Generative models learn the likelihood p(x | y) and prior P(y), then invert with
Bayes; the rule is always the MAP choice ˆy = arg maxk πk p(x | y=k).
Naive Bayes assumes conditional independence, p(x | y) = Qj p(xj | y) — cheap,
fast, and surprisingly accurate (but over-confident; use smoothing).
GDA models each class as a Gaussian: QDA (own Σk) is quadratic, LDA (shared Σ)
is linear.
LDA’s posterior equals σ(wTx + b) — the same logistic form as Session 04, derived
not fit.
Generative wins with little data; discriminative wins with plenty.
Next: beyond linear boundaries — features, kernels, and trees.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 37 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 38

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
References
K. P. Murphy (2022). Probabilistic Machine Learning: An Introduction. MIT Press. (Ch. 9
discriminant analysis; Ch. 10 naive Bayes & generative classifiers.)
T. Mitchell (1997). Machine Learning. McGraw-Hill. (Ch. 6 Bayesian learning & naive Bayes.)
C. M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. (Ch. 4 linear models for
classification; §4.2 probabilistic generative models.)
T. Hastie, R. Tibshirani, J. Friedman (2009). The Elements of Statistical Learning, 2nd ed.
Springer. (§4.3 LDA & QDA.)
A. Y. Ng & M. I. Jordan (2001). On discriminative vs. generative classifiers. NeurIPS.
Jan Hurtado (UTEC) Generative & Probabilistic Models September 7, 2026 38 / 39
```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


#### Página física 39

**Transcripción técnica de la diapositiva (extracción de texto):**

```text

```

**Visual:** curvas de densidad describen p(x|y), y las regiones/fronteras reflejan comparación posterior con Bayes; las flechas muestran el flujo de inferencia, no causalidad.


### 5. ML_Course_Neural_Networks_I_Final.pdf — 53 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 12 —
Neural Networks I
The intuitive version: from one straight line to
a network that learns
Jan Hurtado
UTEC
October 29, 2026
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Let’s start with four points data: the four corners
Example: your turn (1 minute)
Take a sheet: it will hold today’s network. Draw the
four corners of a square: (0, 0) and (1, 1) are circles,
1 (0, 1) and (1, 0) are squares. Now draw one straight
line with both squares on one side and both circles
on the other.
x2
Two lines, or a curve? Hold on to that. Within twenty
minutes we prove that nobody can do it with one line;
0 then we build, by hand, the smallest machine that can.
Key idea
0 1
That machine is a neural network: simple units that
x1
each draw one line, stacked so that together they do
what no single line can.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 2 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Today: from one line to a network
one neuron the wall a hidden layer learning
a weighted sum, bent: four points no line can moves the points until every weight’s slope in
one straight line split one line works one backward sweep
session 04’s logistic proved in one line of built by hand, twice backpropagation, by
regression algebra hand
Intuition
Sessions 02–04 fitted one weighted sum; sessions 08–09 beat straight lines with trees. Today: a
weighted sum of weighted sums. By the end you will know which output, which bend, how wide and
how deep.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 3 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Our examples, one role each
the four corners the ten points
the wall, nets A and B, one line's limit; the tent three scores
backprop the lottery what depth buys softmax by hand
a smooth curve four clouds breast cancer (real data) digits (real data)
what one layer can draw watch it learn yes or no one of ten
Top row: small examples worked by hand. Bottom row: made-up data to see what a network does, and real data
for the last frames. A small badge in the top-right corner of each frame names the data it uses.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 4 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
A neuron: multiply, add, bend
A neuron does three things: multiply the inputs by
weights, add them up with a bias, bend the sum: +b
z = w1x1 + w2x2 + b, a = φ(z) x1
w1 ϕ
(φ, the bend, is also called the activation.)
z
Your turn (30 seconds): w = (2, −1), b = −1, and a
the bend σ(z) = 1/(1 + e−z). Compute z at
A = (1, 1), B = (2, 1) and C = (0, 1). Which are
called 1 (σ > 2)?1 w2 x2
z = 0, 2, −2, so σ = 0.5, 0.88, 0.12. A sits exactly
on the boundary 2x1 −x2 −1 = 0; B is called 1, C is
called 0. multiply add bend
Key idea
With φ = σ this is session 04’s logistic regression, P(y=1 | x) = σ(wTx + b): same formula, same weights, a
new name and a picture. What is new today is stacking them.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 5 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
One neuron draws one straight line data: the ten points
Predict first: session 08’s ten points, which no
straight line separates. How many can one neuron
10 P7 get right?
P3 P6
8 P10
6
x2 P9
P1
4 P4 P5
2
P2 P8
0
0 2 4 6 8 10
x1
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 6 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
One neuron draws one straight line data: the ten points
best line: 9 of 10 Predict first: session 08’s ten points, which no
straight line separates. How many can one neuron
10 P7 get right?
P3 P6 Session 04’s fit (scikit-learn’s default): 7 of 10, with
8 P10 P5, P6 and P8 wrong. Its ten probabilities and our
6 σ(w · x + b) agree to within 10−15: it is the same
x2 P9 function. P1
4 P4 P5 Is 7 the best a line can do? No: 6x1 + 5x2 > 63.5
gets 9 of 10. P3 and P5 score 63, P6 scores 64;
2 only P8 = (7, 2) is wrong. The fit makes the
P2 P8 probabilities likely; it does not count mistakes. And
0
0 2 4 6 8 10 no line gets all 10. (Session 09’s best stump, the line
x1 x1 = 3.5, got 8 of 10; x2 = 5.5 ties.)
Watch out
One neuron = one straight boundary. σ(z) > 12 exactly when z > 0, and z = 0 is a line: the bend bends the
probability, never the boundary.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 6 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Four corners: what can one neuron do? data: the four corners
x1 x2 y Predict first: the best a single neuron can do
on these four? 0 0 0 same
0 1 1 different A computer tried 274,625 neurons: every
1 0 1 different
w1, w2, b from −4 to 4 in steps of 8.1 The best 1 1 0 same
get 3 of 4. Not one gets 4 of 4.
XOR: are the two inputs different?
Of the 16 ways to label four corners, one
274,625 neurons: neuron can produce 14: all but XOR and its
right answers out of 4 opposite. And session 04’s fit returns
194,545 w = (0, 0), b = 0: “probability 0.5” at all four,
which is 2 of 4.
40,304 39,776
0
1 of 4 2 of 4 3 of 4 4 of 4
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 7 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why no line can: add the diagonals data: the four corners
Example: your turn (2 minutes) 1 1 2
On your sheet: pick your own w1, w2, b (small integers).
Compute f = w1x1+w2x2+b at the four corners. Add each
x2
diagonal: f (0, 0) + f (1, 1) and f (0, 1) + f (1, 0). Compare
with your neighbor.
0 1 4
0 x1 1
f (0, 0) + f (1, 1) = b + (w1 + w2 + b) = w1 + w2 + 2b
f (0, 1) + f (1, 0) = (w2 + b) + (w1 + b) = w1 + w2 + 2b 1 + 2 = 3
1 + 4 = 3
XOR needs f < 0 at (0, 0) and (1, 1): a negative sum; and
Key idea
f > 0 at (0, 1) and (1, 0): a positive sum. Equal and of
No line, so no neuron, can do XOR: 3
opposite signs: impossible.
of 4 is a ceiling, proved. No training,
no data, no patience will move it.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 8 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Not only lines: any model that adds data: the four corners
The proof only used that x1 and x2 contribute separately. So for
any f = g1(x1) + g2(x2):
1 2.5 0.5
f (0, 0)+f (1, 1) = g1(0)+g2(0)+g1(1)+g2(1) = f (0, 1)+f (1, 0).
A stump asks about one feature, so a sum of stumps (session x2
09’s boosting with stumps) is additive. Example:
F = 0.5 + 2 · [x1 > 2]1 −3 · [x2 > 2]1 gives 0.5, −2.5, 2.5, −0.5. 0 0.5 2.5
The diagonals sum to 0 and 0.
0 x1 1
0.5 + ( 0.5) = 0
2.5 + 2.5 = 0
Key idea
No model that adds up per-feature pieces can do XOR, however many stumps it adds. To break the identity
the inputs must interact: a tree’s second question inside the first (session 08), a product x1x2, or, today, a
hidden layer.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 9 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Measured: boosting on the corners data: the four corners
optional
Session 09’s gradient boosting, run on the four corners.
stumps, 1,000 rounds: 2 of 4 right depth 2, one round: 4 of 4 right
1 0 0 1 0.2 0.2
x2 x2
0 0 0 0 0.2 0.2
0 x1 1 0 x1 1
With stumps (depth 1), 1,000 rounds: the score is 0 at all four corners, so 2 of 4 right, forever.
With trees of depth 2: one round gives −0.2, 0.2, 0.2, −0.2, so 4 of 4.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 10 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: the wall data: the four corners
Example: check yourself
(a) “XNOR”: same →1, different →0. Can one neuron
do it?
(b) f = x1 + x2 −2x1x2 −0.5 at the four corners: does it
do XOR? Which term breaks the identity?
(c) The best a single stump can do on XOR? ?
(a) No. XNOR needs f > 0 at (0, 0) and (1, 1) and f < 0 at
(0, 1) and (1, 0): two equal sums of opposite signs, the same
proof with the signs swapped. (b) −0.5, 0.5, 0.5, −0.5: yes.
x1x2 mixes the inputs, so f is not g1(x1) + g2(x2). (c) 2 of 4:
every cut leaves one 0 and one 1 on each side.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 11 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Three gates, wired: net A data: the four corners
Each gate is one neuron with a step bend: 1 if z > 0,
else 0.
gate w b fires when OR
x1 1
OR (1, 1) −0.5 at least one input is 1 b = 0.5
NAND (−1, −1) 1.5 not both are 1 1 1
AND (1, 1) −1.5 both are 1 AND
x1 XOR x2 = (x1 OR x2) AND (x1 NAND x2) b = 1.5
1 1At least one, but not both.
Net A: hidden unit 1 = OR, hidden unit 2 = NAND, NAND
x2 1
output = AND of the two. Nine numbers, each a whole b = 1.5
number or a half-integer.
Example: your turn (1 minute): be net A
Left half of the room: you are OR. Right half:
NAND. Front row: AND, on the two halves’ an-
swers. I call a corner; each half answers 0 or 1, then
the front row answers. First (1, 1), then (0, 1).
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 12 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The hidden layer moved the points data: the four corners
x z(1) h z(2) = h1 + h2 −1.5 ˆy y
(0, 0) (−0.5, 1.5) (0, 1) −0.5 0 0
(0, 1) (0.5, 0.5) (1, 1) 0.5 1 1
(1, 0) (0.5, 0.5) (1, 1) 0.5 1 1
(1, 1) (1.5, −0.5) (1, 0) −0.5 0 0
So one straight line finishes the job, the AND unit before after
h1 + h2 = 1.5: z(2) = −0.5, 0.5, 0.5, −0.5, so
ˆy = 0, 1, 1, 0. Four of four, and no z is 0, so no ties. 1 1
x2
0 (NAND) (0,1) and (1,0) both land here Key idea h2
0 x1 1 0 The layer did not bend the line. It moved the points,
and in the new coordinates one straight line is enough. 0 h1 (OR) 1 Session 07’s kernels moved points too, with features
we chose; a network learns its own.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 13 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
But net A cannot learn data: the four corners
Break net A: output bias −0.5 instead of −1.5. It net A's loss as one number moves
now says 1 at all four corners, so 2 are wrong: the
loss L = 21 P(ˆy −y)2 = 1. you are here: slope 0
Nudge any of its 9 numbers by 0.001, or even by 1
0.49: the loss stays exactly 1. All nine slopes are L
0. The step is flat everywhere, and jumps at 0. loss
net A: 1.5
0
3 2 1 0
output bias b (2)
Key idea
Gradient descent follows slopes, and a slope of 0 says nothing about which way to go. We had to
know the answer to write net A down. To learn, a unit needs a bend with a slope.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 14 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bends with a slope
bend formula output range slope
sigmoid σ 1/(1 + e−z) (0, 1) σ(1 −σ), at most 14
tanh 2σ(2z) −1 (−1, 1) 1 −tanh2, at most 1
ReLU max(0, z) [0, ∞) 0 or 1 (none at 0)
step (net A) 1 if z > 0, else 0 {0, 1} 0 (none at 0)
Your turn (30 seconds): σ(0)? And its slope sigmoid tanh ReLU
σ(0) 1 −σ(0) ? 1 1 4
1 2 and 4:1 the sigmoid’s slope is at most 0.25, bend
reached at z = 0. Remember this number. 0 1 0
4 0 4 4 0 4 4 0 4
Key idea
max 0.25 1 1Hidden layers: ReLU is the default: cheap, with 0.25
slope 1 whenever it is on. The output’s bend is slope
chosen by the task (after the break). 0 0 0
4 0 4 4 0 4 4 0 4
z z z
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 15 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Net B: a ReLU network in integers data: the four corners
W (1) = ( 11 11 ), b(1) = −10 , w(2) = (1, −2), b(2) = 0, ReLU
φ = ReLU, ˆy = w(2) · h.
h1
x z(1) h = ReLU(z(1)) ˆy y x1 1
b = 0
(0, 0) (0, −1) (0, 0) 0 0 1 (0, 1) (1, 0) (1, 0) 1 1 1
ˆy (1, 0) (1, 0) (1, 0) 1 1
(1, 1) (2, 1) (2, 1) 0 0 b = 0
1Your turn (1 minute): the corner (1, 1). 2
2 −2 = 0: XOR exactly, with no threshold. ˆy itself is h2
0 or 1. x2 1 b = 1
Three pre-activations are exactly 0, where ReLU has its kink;
libraries use slope 0 there. Harmless here: those rows have no
error.
Key idea
Nets A and B are different, and both exact. Swapping a net’s two hidden units gives a third. The solution is
not unique.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 16 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: hidden layers data: the four corners
Example: check yourself
(a) Where do the four corners land in net B’s hidden
space? Is the identity broken?
(b) Swap net B’s two hidden units. Write its new b(1) and ?
w(2); does it still do XOR?
(c) Which of net A’s nine numbers would gradient
descent change first?
(a) (0, 0), (1, 0), (1, 0), (2, 1); the sums are (2, 1) against (2, 0),
so it is broken. (b) b(1) = (−1, 0), w(2) = (−2, 1): 0, 1, 1, 0, yes,
a third solution. (c) None: every slope is 0.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 17 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The layer rule: mix, then bend data: the four corners
Net B as matrices:
z(1) = W (1)x + b(1) = ( 11 11 )( 11 ) + −10 = ( 21 ) (mix); W (1)
2 x1 h1 h = ReLU(z(1)) = ( 1 ) (bend);
ˆy = w(2) · h = 2 −2 = 0. 1 1 h1
ˆy
Every layer, the same two steps: 1 1 h2
z(l) = W (l)a(l−1)+b(l) (mix), a(l) = φ(z(l)) (bend) x2 h2 rows are
destinations
from a(0) = x to ˆy = a(D) (D layers).
W (l) has one row per unit of layer l and one column
per unit below: rows are destinations. W ij(l) is into 62 numbers
unit i, from unit j. 3-5-5-2
Count: one weight per arrow, one bias per unit. Nets
A and B: 4 + 2 + 2 + 1 = 9. 3-5-5-2:
15 + 5 + 25 + 5 + 10 + 2 = 62. A 28 × 28-pixel digit
reader with 128 hidden units:
784 · 128 + 128 + 128 · 10 + 10 = 101,770.
Symbols today: L is always the loss; D the number of layers; σ only the sigmoid, φ any bend.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 18 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Drop the bend: what is left? data: the four corners
Your turn (1 minute): drop net B’s bend (h = z(1)) and
multiply out: ˆy = w(2) · (W (1)x + b(1)) + b(2) = 1 ?1 ?0
(w(2)W (1)) x + (w(2) · b(1) + b(2)). Does it still do XOR?
x2
0 ?2 ?1
0 x1 1
2 + 0 = 2
1 + 1 = 2
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 19 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Drop the bend: what is left? data: the four corners
Your turn (1 minute): drop net B’s bend (h = z(1)) and
multiply out: ˆy = w(2) · (W (1)x + b(1)) + b(2) = 1 1 0
(w(2)W (1)) x + (w(2) · b(1) + b(2)). Does it still do XOR?
w(2)W (1) = (1, −2)( 11 11 ) = (−1, −1), and x2 w(2) · b(1) + b(2) = 2. So ˆy = −x1 −x2 + 2: one line. At the
corners: 2, 1, 1, 0; the diagonals 2 = 2. The wall is back.
Always: 0 2 1
W (2)(W (1)x+b(1))+b(2) = (W (2)W (1)) x+(W (2)b(1) + b(2))
0 x1 1 | one {zmatrix } | one{zvector }
A stack of layers without bends collapses into one linear 2 + 0 = 2
layer, however tall. Measured: a random 2-8-8-1 stack 1 + 1 = 2
without bends has 105 numbers in and 3 out (two weights
and a bias), agreeing to 13 decimals. Key idea
The bend is not decoration:
without it depth buys nothing, and
the network is session 02’s linear
regression in disguise.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 19 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
With the bend, depth multiplies data: the tent
Two ReLU units make a tent: k = 1: 2 pieces k = 2: 4 pieces
t(x) = 2 ReLU(x) −4 ReLU(x −12). Your turn 1 1
(1 minute): t at 0, 14, 12, 4,3 1.
x 0 14 21 34 1
0 0
t(x) 0 0.5 1 0.5 0 0 ½ 1 0 ½ 1
Up, then down: t maps [0, 1] onto [0, 1]. So k = 3: 8 pieces k = 4: 16 pieces
t(t(x)) runs the tent twice: at 0, 8,1 14, 38, 21 it 1 1
gives 0, 12, 1, 12, 0. That is four pieces.
Each fold adds a layer of 2 units and doubles
the pieces: k folds = 2k units, 2k pieces. 0 0
0 ½ 1 0 ½ 1
One hidden layer: each unit adds at most one
kink, so 2k pieces need at least 2k −1 units. 12 Key idea
folds: 24 units in 12 layers, against at least
Width adds pieces; depth multiplies them.
4,095 in one layer.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 20 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: layers data: the four corners
Example: check yourself
(a) Count the numbers of a 2-3-1 and of a 4-8-3 network.
(b) Net A with its steps removed (φ(z) = z): what does
it output? ?
(c) 5 folds of the tent: units, pieces, and a lower bound
for one layer?
(a) 6 + 3 + 3 + 1 = 13; 32 + 8 + 24 + 3 = 67. (b)
(x1 + x2 −0.5) + (−x1 −x2 + 1.5) −1.5 = −0.5 at every corner:
a constant, 2 of 4. (c) 10 units, 32 pieces; one layer needs at
least 31.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 21 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 24

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Break — through the wall, by hand data: the four corners
one neuron = one line One neuron draws one line. XOR needs its
two diagonals to disagree, and no line, no
the wall: f (0, 0)+f (1, 1) = f (0, 1)+f (1, 0)
additive model, can do that.
a hidden layer moves the points (net A) A hidden layer moves the points; in (h1, h2)
one line finishes the job. Without a bend the
a bend with a slope (ReLU: net B) stack collapses back to a line.
Depth multiplies: k folds of 2 ReLU units z = W a + b, a = φ(z)
make 2k pieces.
next: what one layer can draw;
heads and losses; backpropagation
Key idea
After the break: what one wide layer can draw, and what that does not promise; the output and its
loss; how a network finds its own weights (backpropagation, by hand on net B).
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 22 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 25

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two S-curves, subtracted
σ s(x −0.3) −σ s(x −0.5) , s the steepness.
1
Predict first: two S-curves switching on at 0.3
and at 0.5, subtracted. What shape?
σ(50(x −0.3))
σ(50(x −0.5)) ?
difference
0
0 0.3 0.5 1
x
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 23 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 26

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Two S-curves, subtracted
σ s(x −0.3) −σ s(x −0.5) , s the steepness.
1
Predict first: two S-curves switching on at 0.3
and at 0.5, subtracted. What shape?
A bump: up at 0.3, down at 0.5. σ(50(x −0.3))
Steeper units, a sharper bump, which needs big
σ(50(x −0.5)) weights:
difference s 5 20 50 200
height 0.24 0.76 0.99 1.00
width at half height 0.72 0.24 0.20 0.20 0
0 0.3 0.5 1
As a network: two hidden units (input weights s x
and s, biases −0.3s and −0.5s), output weights
+1 and −1. Two units = one bump.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 23 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 27

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Many bumps: any curve data: a smooth curve
H = 2 bumps (4 units)
1 the curve, sin 2 x
the staircase
0
1
0 0.25 0.5 0.75 1
x
Give bump j the curve’s height at its middle: a staircase. H = 2 bumps (4 units): largest error
0.99.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 24 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 28

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Many bumps: any curve data: a smooth curve
H = 8 bumps (16 units)
1 the curve, sin 2 x
the staircase
0
1
0 0.25 0.5 0.75 1
x
H = 8 bumps (16 units): 0.37.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 24 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 29

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Many bumps: any curve data: a smooth curve
H = 32 bumps (64 units)
1 the curve, sin 2 x
the staircase
0
1
0 0.25 0.5 0.75 1
x
H = 32 bumps (64 units): 0.090. Your turn (30 seconds): predict H = 64.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 24 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 30

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Many bumps: any curve data: a smooth curve
H = 32 bumps (64 units)
1 the curve, sin 2 x
the staircase
0
1
0 0.25 0.5 0.75 1
x
0.042: double the units, halve the error. A step misses the curve by at most (steepest slope) ×
(half a step) = 2π · 2H1 = π/H.
Key idea
Universal approximation, as a construction: one hidden layer with enough units draws any continuous curve
on a closed interval [a, b], as closely as you like.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 24 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 31

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The theorem, stated properly data: a smooth curve
Let the bend φ be continuous and not a polynomial, and f continuous on a closed, optional
bounded set K. Then for every ε > 0 some network with one hidden layer and
finitely many units stays within ε of f everywhere on K (Leshno, Lin, Pinkus and
Schocken 1993; earlier for sigmoids, Cybenko 1989, and for bounded bends, Hornik
1991).
Why not a polynomial: a network built from a polynomial bend is itself a polynomial of fixed
degree, which cannot follow every curve. Why bounded: on an unbounded region a curve can
keep wiggling, and finitely many bumps run out.
H = 32 bumps (64 units)
1 the curve, sin 2 x
the staircase
0
1
0 0.25 0.5 0.75 1
x
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 25 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 32

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What the theorem does not promise data: a smooth curve + noise
2 units 32 units 2
0
(leaves the picture)
2
0 0.5 1 0 0.5 1
x x
How many units: none is given. Our staircase’s error falls only like 1/H: ten times closer costs
about ten times the units.
That training finds it: we test that before the end of today. Predict first: on these 20 noisy
points, which follows the dashed true curve better, 2 units or 32?
That it predicts new points: 20 noisy points of the curve (medians of 10 samples; the picture
shows one).
2 units: error 0.24 on the points, 0.22 off the true curve.
32 units: 0.13 on the points, half the true curve’s own 0.27: it learned the noise, and is more
than 1 off the curve.
Worse than 2 units on 8 of 10 samples, by more than 0.3; on the other 2 it found the
same fit.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 26 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 33

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: drawing with units data: a smooth curve
Example: check yourself
(a) Write the two units of a bump on [0.2, 0.6] with
s = 100.
(b) The staircase with H = 16 misses by about 0.19.
About how many bumps for 0.02? ?
(c) A huge network fits its training points perfectly but
predicts new ones badly. Is that a broken theorem?
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 27 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 34

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Three heads, three losses data: three scores
the answer is output unit loss from
a number identity: ˆy = z squared error 12 (ˆy −y)2 session 02
yes or no sigmoid: p = σ(z) cross-entropy − y ln p + (1 −y) ln(1 −p) session 04
one of K softmax: pk = ezk /Pj ezj cross-entropy −ln pc (c the true class) session 04
Your turn (1 minute): softmax of
z = (ln 4, ln 2, 0). z ez: sum 7 p = ez/7
1.386 4 4/7 ez = (4, 2, 1), sum 7, so
p = ( 7,4 7,2 17) = (0.571, 0.286, 0.143). If class 1 is
true, the loss is −ln 47 = ln 47 = 0.560. 0.693 2 2/7
1 1/7
Key idea 0
The body builds features; the head is a model you already 1 2 3 1 2 3 1 2 3
know, session 02’s or session 04’s.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 28 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 35

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Predicted minus actual data: three scores
ˆy class by class L = −ln pc = −zc + ln Pj ezj −y, sum = 0 +2/7
∂L ezj +1/7 = −[j=c] + = pj −yj =⇒ ∂L∂z = ˆy −y
∂zj Pk ezk
The same ˆy −y for sigmoid + cross-entropy (session 04: the
two p(1 −p) factors cancel) and for identity + squared error
(the slope of 12(ˆy −y)2 is ˆy −y). 3/7
1 (true) 2 3
Key idea
Predicted minus actual: with a matched head and loss, the output’s error signal is always ˆy −y.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 29 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 36

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which loss pushes harder? data: one confident unit
Predict first: a sigmoid unit says 0.0025 0.9975
(z = −6), but the answer is 1. Which loss pushes 100 cross-entropy it harder: cross-entropy or squared error?
z|
Cross-entropy: ˆy −y = −0.9975. Squared error: 10 1
L/
(ˆy −y) σ′(z) = −0.0025. That is 405 times | squared
weaker: σ′ is tiny whenever the unit is confident, 10 2 error
and so also when it is confidently wrong. push 0.0025
Steps of size 1 until it answers right (p > 12): 10 3
cross-entropy 7, squared error 426. the unit says 0.0025
10 4
8 6 4 0 4 8
z (answer y = 1)
Watch out
Choose the loss with the output: cross-entropy for sigmoid and softmax, squared error for the identity.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 30 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 37

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: heads and losses data: three scores
Example: check yourself
(a) Logits (0, 0, 0): softmax? loss? ∂L/∂z if class 1 is
true?
(b) Which head and loss for tomorrow’s temperature; spam or not; which of 10 digits? ?
(c) Logits (ln 3, 0, 0): the probabilities, and the loss if
class 1 is true?
(a) ( 13, 13, 13); ln 3 = 1.099; (−23, 13, 13). (b) identity + squared
error; sigmoid + cross-entropy; softmax + cross-entropy. (c)
(0.6, 0.2, 0.2); ln 53 = 0.511.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 31 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 38

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which way should each number move? data: the four corners
Break net B: w(2) = (1, −1) instead of (1, −2). At
x = (1, 1): z(1) = h = (2, 1), ˆy = 2 −1 = 1, but
y = 0, so L = 12(ˆy −y)2 = 0.5. The other three x1 = 1 h1 = 2
corners are still right. Predict first: which of its 9
= 0 numbers should move, and which way? 1 y
Find out by nudging. w 2(2) from −1 to −0.999: ˆy = 1
ˆy = 1.001, L = 0.5010005, slope ≈1.0005 ≈1.
w 1(2) + 0.001: slope ≈2.002 ≈2. A positive slope 1
means: make that number smaller. The weight we did the one
not break has the larger slope. x2 = 1 h2 = 1 we broke
Nudging costs one forward pass per number. 10 layers
of 100 units: about 10 × 100 × 100 = 100,000 weights,
so 100,000 passes of 100,000 operations: 1010.
Key idea
Backpropagation gets all of them for about the price of two passes (2 × 105): the chain rule, organized so
that nothing is computed twice.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 32 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 39

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The error signal δ data: the four corners
Give every unit one number, its blame: δ = ∂L/∂z. It
says how much the loss changes if this unit’s input z is unit j unit i
nudged.
Wij
A weight Wij enters only through zi = Pm Wimam + bi, aj zi
so ∂L/∂Wij = (∂L/∂zi)(∂zi/∂Wij) = δi aj (the chain
rule: the unit’s blame × what came in on that wire)
δi = L/ zi
and ∂L/∂bi = δi. All at once:
∂L ∂L
(l) = δ(l) a(l−1)T, = δ(l) L/ Wij = δi × aj ∂W ∂b(l)
At the output, the cancellation: δ(D) = ˆy −y. Net B:
δ(2) = 1 −0 = 1, so ∂L/∂w(2) = δ(2)h = (2, 1).
Key idea
Every gradient is a blame times an input. What is left: the blame of the hidden units.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 33 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 40

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Passing the blame backward
forward x W (1) + b (1) z(1) ϕ a(1) W (2) + b (2) ˆy loss L · ·
backward δ (1) ⊙ϕ′(z(1)) L/ a(1) ×W (2) ⊤ δ (2) = ˆy −y
L/ W (1) = δ (1)x ⊤ L/ W (2) = δ (2)a(1) ⊤
Hidden unit j feeds every unit i above it through Wij, so its output’s blame collects theirs:
∂L/∂aj = Pi δi ∂zi/∂aj = Pi Wijδi, that is W (l+1)Tδ(l+1).
Then back through its own bend, times the bend’s slope:
δ(l) = W (l+1)Tδ(l+1) ⊙φ′ z(l) (⊙: unit by unit)
Two kinds of step, two factors: across a mix, W T; across a bend, φ′. Mixing them up is the
classic bug.
Every sigmoid layer multiplies the blame by σ′ ≤0.25, so through 10 of them it shrinks by a factor
of at least 410 = 1,048,576 (before the weights act): the vanishing gradient. ReLU’s slope is 1
where the unit is on: it removes this factor, not the problem, since W T still scales it.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 34 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 41

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
By hand: backpropagation on net B data: the four corners
W (1) = 11 11 , b(1) = (0, −1), w(2) = (1, −1) (broken), b(2) = 0; x = (1, 1), y = 0; L = 21 (ˆy −y)2.
step value x = (1,1) L/ W (1), rows: (1,1), ( 1, 1)
forward z(1) = W (1)x + b(1); h = ReLU(z(1)) (2, 1); (2, 1) mix
ˆy = w(2) · h + b(2); L 1; 0.5 z(1) = (2,1) δ (1) = (1, 1)
φ′(z(1)): both units on (1, 1) bend (1,1)
1 δ(2) = ˆy −y 1 h = (2,1) L/ h = (1, 1)
2 ∂L/∂w(2) = δ(2)h (2, 1) mix × (1, 1)
3 ∂L/∂b(2) = δ(2) 1 ˆy = 1 δ (2) = 1
4 δ(1) = (w(2)δ(2)) ⊙φ′(z(1)) (1, −1) ˆy (1) 1 1 loss −y5 ∂L/∂W = δ(1)xT −1 −1 L = 0.5 L/ w(2) = (2,1)6 ∂L/∂b(1) = δ(1) (1, −1)
Your turn (2 minutes): rows 4–6 on your sheet.
Key idea
The answer to the prediction: all nine numbers move, not only the one we broke, and the biggest slope, 2, is
that of w(2)1 , which we never touched (blame × input, h1 = 2); the nudges agree (2.002 and 1.0005). Only
(1, 1) has an error, so this one row is the gradient over all four corners.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 35 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 42

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The algorithm, and why it is cheap data: the four corners
1 Forward: compute and keep every z(l) and a(l). Loss nudge every number units ×
L(ˆy, y). layers backpropagation
2 Output: δ(D) = ˆy −y. Backward, l = D −1, . . . , 1: 9×10
δ(l) = (W (l+1)Tδ(l+1)) ⊙φ′(z(l)). 10×3 600
3 Read off ∂L/∂W (l) = δ(l)a(l−1)T, ∂L/∂b(l) = δ(l). Step: 2.03×10
30×5 W ←W −η ∂L/∂W (η, the learning rate: session 02’s 9,000
step size). Net B with η = 0.1, moving only w(2): 10¹
w(2) = (1, −1) −0.1 · (2, 1) = (0.8, −1.1). Why 0.1? 100×10 2×10 50,000× less
Session 13’s first question.
1.82×10¹²
300×15 2.7×10
Key idea
Why it is cheap: each δ(l) is computed once and reused by 102 106 1010 1014
every weight of its layer and by the layer below. The backward operations per step
pass costs about one forward pass, so all the slopes cost about
2 × 105 operations instead of 1010 for nudging, at 10 layers of
100 units.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 36 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 43

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Watch it learn data: four clouds
The loop: random start →forward →loss →backward →step →repeat. One pass over the data is an
epoch. Here 160 points in four clouds, and a 2-4-1 network (4 tanh units, sigmoid output, average
cross-entropy); every step uses all the points, with η = 0.5.
Predict first: what will the
boundary look like at the end?
Epoch 0: 62 of 160 right. The
loss falls from the first step:
0.705 →0.582 by epoch 30 (the
epoch axis is logarithmic).
At epoch 75 all 160 are right (loss
0.300); at epoch 4,000 the loss is
0.001. Two curved cuts made a
checkerboard.
Watch out
Loss, not count: from epoch 10 to 20
the loss fell every step, yet the right
answers fell from 131 to 123.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 37 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 44

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
What did it learn? Its own gates data: four clouds
optional
unit 1 The four hidden units after
cuts off (1,1), an AND 4,000 epochs. Each is one tanh
1 neuron, so each draws one line.
unit 2 Each line cuts off one corner.
cuts off (0,1) Unit 1 fires only at (1, 1): anx2
AND. Unit 4 fires everywhere
unit 3
but (0, 0): an OR. Units 2 and
cuts off (1,0) 0 3 fire everywhere but (0, 1) and
unit 4 (1, 0). The output weighs them
cuts off (0,0), an OR −3.4, −5.8, −6.0, +5.2. It
0 x1 1 invented its own gates, like net
A.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 38 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 45

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Does training always find it? data: the four corners + the ten points
Predict first: nets A and B prove that a 2-unit
XOR the ten points network can do XOR. Train a 2-unit network
(tanh) from 200 random starts: how many reach all 200200 189 197 200 4 of 4?
of 154
fit, 142 95 of 200, about half. With 3 units 154, with 4
95 105 102 units 189. A solution that exists is not always 100 found: where you start matters.that
41 The ten points, 8 units: 10 of 10 from 197 of
30 27 200 starts with standardized inputs, from 102 ofstarts 8
0 200 with raw ones. 4 standardized units (142 of 2 3 4 2 3 4 8 200) beat 8 raw ones.
hidden units
raw inputs standardized
Key idea
We can compute every gradient; using them well is the next question. Session 13 answers four: how big a
step? which examples? where to start? when to stop?
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 39 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 46

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Check yourself: who gets the blame? data: the four corners
Example: check yourself
(a) The correct net B (w(2) = (1, −2)) at x = (1, 1):
every gradient?
(b) w(2) = (1, 0) at x = (1, 1), y = 0: ˆy, δ(2), ∂L/∂w(2), δ(1), ∂L/∂W (1)? ?
(c) Your 2-unit network got 3 of 4 on XOR. Two things
to try?
(a) All 0: no error, no blame. (b) ˆy = 2, δ(2) = 2, (4, 2),
δ(1) = (2, 0), rows (2, 2), (0, 0). Unit 2’s outgoing weight is 0, so
it gets no blame and its weights stay put. (c) Another random
start; more units (3 units: 154 of 200; 4: 189 of 200).
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 40 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 47

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which output, which loss? The evidence data: today’s data
identity sigmoid softmax sigmoid
squared error cross-entropy cross-entropy squared error
a number: a smooth curve 0.03 · · 0.50
error (lower is better)
yes or no: breast cancer 47% 0% 0% 0%
answers outside [0, 1]
one of ten: digits 0.971 0.964 0.960 0.962
test accuracy
one of ten: probabilities? 0.4% 70% 100% 50%
outputs that are probabilities
a confident mistake 1 7 4 426
steps to fix it
The task picks the head: identity for a number, sigmoid for yes/no, softmax for one of K, each with its
matched loss. Accuracy alone cannot tell them apart; squared error on a sigmoid needs 426 steps to fix
what cross-entropy fixes in 7. A sigmoid is never below 0, so on the curve its error is at least about 0.50.
Same body in every column: 16 ReLU units, standardized inputs, 500 full-batch steps of Adam (a refinement of gradient descent that session 13
explains), median of 5 random starts. Error: root mean square over the curve’s 201 points. Test sets: 143 patients, 450 digits. Probabilities: test digits
whose 10 outputs are all in [0, 1] and add to 1 (±0.01). Mistake: one unit at z = −6 (a sigmoid says 0.0025; softmax: logits (−6, 0), both move),
answer 1, steps of size 1. · = cannot: cross-entropy needs answers in [0, 1]; a number is not a class.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 41 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 48

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Bend, width, depth: the evidence data: today’s data
one no bend ReLU ReLU ReLU sigmoid
neuron 2 × 16 2 units 32 units 4 × 8 4 × 16
XOR: the four corners 0 0 45 200 180 0
starts that fit, of 200
the ten points 0 0 38 200 168 65
starts that fit, of 200
a smooth curve 0.447 0.447 0.447 0.018 0.025 0.705
error, median of 5
a zigzag (8 pieces) 0 0 0 6 10 0
starts that nail it, of 60
digits (real data) 0.953 0.956 0.542 0.971 0.913 0.596
test accuracy, median of 3
One wide ReLU layer is green on 4 of 5 rows. No bend scores like one neuron everywhere (the collapse);
2 units are too few; four sigmoid layers stall: on most starts lbfgs quits within 3 steps at a constant
answer, as four sigmoids leave it almost no slope (the vanishing gradient).
scikit-learn networks with the matched head and loss; the ten points and the digits standardized, the corners and the curves already on [0, 1]: lbfgs
(default tolerance) on the four small rows, scikit-learn’s default solver (300 passes) on the digits. Fit = every training point right; error = root mean
square (median of 5 starts); nail = error below 0.01 (zigzag: the tent folded 3 times). One layer of 64 units nails it in 29 of 60; 8 units in one layer
could draw it exactly.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 42 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 49

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Use it when, avoid it when
use it when avoid it when you choose
identity + squared the answer is a number yes/no or classes: 47% of its answers —
error left [0, 1]
sigmoid + yes or no; you want a numbers below 0 or above 1; never a threshold
cross-entropy probability with squared error (426 steps against 7) ( 12 )
softmax + one of K classes: K several labels can be true at once (then K
cross-entropy probabilities that add to 1 K sigmoids)
ReLU hidden units the default for hidden layers too few units (2: 45 of 200 on XOR) how many
sigmoid or tanh one shallow layer many layers: σ′ ≤0.25 per layer, and —
hidden units tanh′ fades once a unit saturates
width add units while held-out it fits the noise (32 units: more than 1 units
error falls off the curve; 2 units: 0.22)
depth one layer needs far more small data: one wide layer did as well layers
(tent: 24 against ≥4,095) or better (digits: 0.971 against 0.913)
Watch out Key idea
Standardize the inputs first: on the ten points 8 Start with one hidden layer of ReLU units, the head
units fit from 197 of 200 starts standardized, from the task asks for and its matched loss; widen it while
102 of 200 raw. Never a hidden layer without a the error on held-out data falls (session 03); then
bend: it collapses. try several starts.
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 43 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 50

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Which one should I use?
first: standardize the inputs
a number
identity + squared error
yes or no
what is the answer? sigmoid + cross-entropy
one of K
softmax + cross-entropy
size: one layer of tens of units,
hidden layers: ReLU
wider while held-out error falls
beyond today: how to step, deeper: only when one layer would need far
on which examples, from more units (the tent), and with lots of data
where, until when (session 13)
train from several random starts;
judge on data it never saw
Then look: does it fit the training data, and does it predict new data?
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 44 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 51

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
In practice: the knobs, and the code data: digits (real data)
hidden_layer_sizes=(32,): the
width; (16, 16) is two layers. from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
activation: "relu" (default),
from sklearn.pipeline import make_pipeline
"logistic" (σ), "tanh", "identity"
(no bend: collapses).
net = make_pipeline(
The head comes with the class. StandardScaler(), # one scale
MLPClassifier uses a sigmoid (2 MLPClassifier(
classes) or softmax (K) with hidden_layer_sizes=(32,), # one layer
cross-entropy; MLPRegressor uses the activation="relu", # the bend
identity with squared error. random_state=0)) # the start
random_state: the start; try several. net.fit(X_train, y_train)
print(net.score(X_test, y_test)) # 0.98 on the digits
StandardScaler first.
mlp = net[-1]
print(mlp.out_activation_) # ’softmax’ (10 classes)
print(mlp.coefs_[0].shape) # (64, 32): W transposed
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 45 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 52

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Cheat sheet
the idea formula watch out
neuron multiply, add, bend: one line a = φ(w · x + b) σ(z) > 12 ⇐⇒z > 0
the wall no line, no additive model, can f (0, 0) + f (1, 1) = fitted ̸= best possible line
do XOR f (0, 1) + f (1, 0)
layers mix, then bend; they move the z(l) = W (l)a(l−1) + b(l), rows are destinations; no
points a(l) = φ(z(l)) bend means a collapse; the
step has no slope
H bumps = 2H units, error exists ̸= found ̸= predicts
width, depth ∼1/H; k folds = 2k units, 2k new points
pieces
output + identity + squared error; sigmoid δ(D) = ˆy −y never sigmoid + squared
loss or softmax + cross-entropy error
backprop blame flows back along the same δ(l) = W T across a mix, φ′ across a
wires (W (l+1)Tδ(l+1)) ⊙φ′(z(l)); bend; σ′ ≤0.25
∂L/∂W (l) = δ(l)a(l−1)T
Key idea
A network is session 04’s neuron, stacked, with a bend between layers; backpropagation gives every slope in
two passes. Next session: how big a step? which examples? where to start? when to stop?
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 46 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


#### Página física 53

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Exit ticket
1. Line; moves. 2. 3 · 4 + 4 + 4 · 2 + 2 = 26. 1 One neuron draws one straight ; a
3. z = 1, a = 1, ˆy = 3, L = 2. hidden layer the points.
δ = ˆy −y = 2, so ∂L/∂v = 2 · 1 = 2. The
2 How many numbers does a 3-4-2 network
blame on a is 3 · 2 = 6; ReLU’s slope is 1,
have?
so ∂L/∂w = 6 · 1 · x = 6 · 2 = 12.
3 x = 2 →z = wx + b (w = 1, b = −1) 4. (a) identity + squared error; (b) sigmoid
→a = ReLU(z) →ˆy = v a (v = 3); y = 1, + cross-entropy; (c) softmax +
L = 2(ˆy1 −y)2. Find ˆy, L, ∂L/∂v, ∂L/∂w. cross-entropy.
4 Which output and loss: (a) the price of an
apartment; (b) rain tomorrow or not; (c)
which of 5 flowers?
?
Jan Hurtado (UTEC) Neural networks, intuitively October 29, 2026 47 / 49
```

**Visual:** círculos son neuronas, flechas son conexiones ponderadas, Σ suma ponderada y φ activación. Colores de clases muestran la separabilidad; no confundir color con signo de un peso sin leyenda.


### 6. ML_Course_Kernels.pdf — 23 páginas


#### Página física 1

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lecture 07 — Kernel Methods
The kernel trick, common kernels, and kernel-
ized regression
Jan Hurtado
UTEC
July 26, 2026
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 2

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Roadmap
1 The hook: inner products only
2 Feature maps and the trick
3 What counts as a kernel
4 The kernel zoo
5 Tuning complexity
6 Representer theorem and kernel ridge
7 Wrap-up
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 2 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 3

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Key terms (for newcomers)
Feature map ϕ: sends each input to a Polynomial kernel: (c + xTx′)d; degree d sets
(possibly huge) feature vector. the feature richness.
Kernel K(x, x′): a similarity equal to the inner RBF / Gaussian kernel: exp(−γ∥x −x′∥2);
product ϕ(x)Tϕ(x′). an infinite-dimensional feature space.
The kernel trick: compute K directly, never Bandwidth γ: how local the RBF is — large
forming ϕ. γ overfits.
Gram matrix: the n × n matrix Representer theorem: the optimal f is a
Kij = K(xi, xj) of all pairwise kernel values. weighted sum of kernels at the training points.
PSD (Mercer): the property a valid kernel’s Kernel ridge regression: ridge in feature
Gram matrix must have. space, in closed form.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 3 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 4

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Where Session 06 left us
The dual SVM’s decision rule was
f (x) = X αiyi xTi x + b.
i
The data enters only through inner products xTi xj — never the raw coordinates.
So replace each xTi xj with a kernel K(xi, xj) and the same algorithm draws
nonlinear boundaries.
Key idea
A kernel is a drop-in replacement for the dot product. This session is about the kernels
themselves: what they are, which to use, and how the trick reaches beyond the SVM to
regression.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 4 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 5

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Lift the data until it is linearly separable
Input space: no separating line Lifted by φ: a flat plane separates
3
2
8
7
1 6
5
4
x2 0 3
2
1
1 0
3 2
1
0 2 3 2 1 x2 0 1 2 3 3 2 1 x1
3 2 1 0 1 2 3
x1
Concentric rings (left) admit no separating line. The lift ϕ(x) = (x1, x2, x12 + x22 ) raises the inner ring, so
a flat plane separates the classes in 3-D.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 5 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 6

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Kernel = inner product in feature space
A feature map ϕ sends x to a richer space; a kernel returns the inner product
there:
K(x, x′) = ϕ(x)Tϕ(x′).
The trick: a good K is cheap to evaluate even when ϕ is enormous — or
infinite-dimensional.
We never build ϕ; we only ever ask the kernel “how similar are these two points?”
Intuition
Think of K as a similarity score with a guarantee: it equals a dot product in some space.
That guarantee is what lets every inner-product algorithm inherit nonlinearity for free.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 6 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 7

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Example: the quadratic kernel hides its ϕ
In 2-D, expand K(x, x′) = (1 + xTx′)2 with x = (x1, x2):
(1 + x1x1′ + x2x2)2′ = 1 + 2x1x1′ + 2x2x2′ + x12 x1′2 + x22 x2′2 + 2x1x2 x1x′ 2.′
This is exactly ϕ(x)Tϕ(x′) with the explicit map
√ √ √
ϕ(x) = 1, 2 x1, 2 x2, x12 , x22 , 2 x1x2 .
Key idea
The kernel computes that six-dimensional dot product without ever building the six features — one
squared scalar versus six multiplications. The saving explodes as the degree grows.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 7 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 8

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Valid kernels: symmetric and PSD
A function K is a valid kernel iff it is symmetric (K(x, x′) = K(x′, x)) and positive
semi-definite (Mercer).
PSD means: every Gram matrix Kij = K(xi, xj) satisfies cTKc ≥0 for all c.
Equivalently, K is an inner product in some feature space — so a ϕ is guaranteed
to exist, even if we never write it down.
Going deeper (optional)
Closure rules let you build kernels: if K1, K2 are kernels, so are K1 + K2, K1K2, and aK1 for a > 0.
Sums and products of valid similarities stay valid.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 8 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 9

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The kernel is a similarity matrix
RBF Gram matrix Kij = K(xi, xj)
1.0
0.8
class 0
0.6
similarity
0.4
class 1 0.2
0.0
class 0 class 1
RBF Gram matrix Kij with rows/columns sorted by class. Bright within-class blocks = high similarity;
PSD is the algebraic stamp of a genuine inner product.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 9 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 10

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Three kernels you will actually use
Linear: K(x, x′) = xTx′ — the plain dot product; recovers the ordinary linear model.
Polynomial: K(x, x′) = (c + xTx′)d — degree d adds all products up to order d; a global kernel.
RBF / Gaussian: K(x, x′) = exp(−γ∥x −x′∥2) — an infinite-dimensional ϕ; a local bump of
width set by γ.
(One more: the sigmoid kernel tanh(a xTx′ + c), used occasionally; not always PSD.)
Intuition
Polynomial similarity grows as points align (global reach); RBF similarity decays with distance (local
reach). That reach is what shapes the boundary.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 10 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 11

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Why the RBF is infinite-dimensional
Factor out the norms and expand the
exponential as a power series:
∞
∞ (2γ)k (2γ)k RBF = k! (x ⊤x′)k: a mix of all polynomial degrees e 2γ xTx′ = X k = 0 X k! (xTx′)k.
0.25 γ = 1 every degree k has non-zero weight k=0
⇒ infinitely many features
k 0.20 Each (xTx′)k is the degree-k polynomial
γ = 3degree 0.15 kernel, so the RBF mixes every
on polynomial degree at once.
0.10weight
0.05 Key idea
Infinitely many degrees ⇒an infinite-
0.00
0 2 4 6 8 10 12 14 16 dimensional ϕ — yet the kernel still polynomial degree k mixed into the RBF
evaluates in one line. Larger γ shifts
weight to higher degrees: more local,
more complex.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 11 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 12

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Local versus global, pictured
Kernels are similarities: RBF is local, polynomial is global
6
linear x · x′ RBF, small γ (wide)
5 polynomial (1 + x · x′)3 RBF, large γ (narrow)
4
3
x′) 2
K(x, 1
0
1
2
3
3 2 1 0 1 2 3
x (the other point fixed at the marked anchor)
K(x, anchor) as x moves: linear grows without bound, polynomial reaches globally, RBF is a local bump
of width γ.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 12 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 13

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Same data, three kernels, three boundaries
Same SVM, same data --- only the kernel changes
linear polynomial (degree 3) RBF (Gaussian)
(train acc 0.83) (train acc 0.90) (train acc 0.94)
One SVM, one dataset. Linear (left) draws a straight cut; polynomial deg-3 (middle) curves; RBF
(right) wraps tightly. The kernel chooses the hypothesis space.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 13 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 14

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The RBF bandwidth γ controls complexity
RBF bandwidth γ controls complexity
γ = 0.1 γ = 1 γ = 60
too smooth (underfit) (acc 0.84) well chosen (acc 0.93) islands (overfit) (acc 0.99)
Small γ (left): bumps so wide the model underfits. Good γ (middle): a clean boundary. Large γ
(right): tiny bumps overfit into islands.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 14 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 15

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Sweeping γ: from underfit to overfit
As γ grows the RBF bumps shrink: a smooth boundary tightens, then fragments into overfit islands.
(Animated PDF — press play to view.)
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 15 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 16

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Tuning cautions
Watch out
Choose γ and the SVM C by cross-validation (Session 03) — never by training accuracy,
which large γ can drive to a misleading 100%.
Watch out
Standardize features first: the RBF distance ∥x −x′∥mixes all coordinates, so an unscaled
feature dominates. And the Gram matrix costs O(n2) memory — kernels suit small-to-medium
n.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 16 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 17

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The representer theorem
Key idea
For a regularized risk minimized over a kernel space, the optimal f is a weighted sum of
kernels at the training points:
f (x) = X αi K(xi, x).
i
Intuition
However huge the feature space, the solution lives in the finite span of the training data. So
we only ever need kernels between training points and the query — exactly what the trick
gives us cheaply.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 17 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 18

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
The representer theorem, pictured
f (x) = Pi αiK(xi, x) is literally one
The representer theorem, pictured: the fit is a sum of kernel bumps bump per training point, scaled by
= x) αiK(xi, 1.0 fit αi; the fit is their sum. Positive αi
> 0 Xi αi bump,
bump, αi < 0 push the curve up, negative pull it
training data
0.5 down.
y 0.0 Going deeper (optional)
Why only training points? Any
0.5 part of w orthogonal to {ϕ(xi)}
leaves every prediction unchanged
1.0 but adds to the penalty ∥w∥2 — so
one bump: αi K(xi, · ) the optimum has none. The solu-
3 2 1 0 1 2 3
x tion lives in the finite span of the
data.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 18 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 19

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Kernel ridge regression: a closed form
Ridge regression done in feature space. By the representer theorem and a short
derivation:
α = (K + λI)−1y, f (x) = X αi K(xi, x).
i
Going deeper (optional)
Minimize ∥y−Kα∥2 +λ αTKα over α; setting the gradient to zero gives (K +λI)α = y. One solve,
no iteration — nonlinear regression in closed form.
Same idea kernelises SVR, kernel PCA, and Gaussian processes.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 19 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 20

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Linear underfits; kernel ridge fits the curve
Linear ridge: underfits Kernel ridge (RBF): fits the curve
1.5 data data
true function true function
1.0 fit fit
0.5
y 0.0
0.5
1.0
1.5
3 2 1 0 1 2 3 3 2 1 0 1 2 3
x x
Left: linear ridge can only draw a line through a sine curve. Right: RBF kernel ridge follows it smoothly
— nonlinear fit, one matrix solve.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 20 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 21

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
Summary
Key idea
Every inner-product algorithm becomes nonlinear by swapping xTi xj for a kernel
K(xi, xj) = ϕ(xi)Tϕ(xj).
The trick: compute K directly, never forming ϕ — even when ϕ is infinite-dimensional
(RBF).
Valid kernels are symmetric and PSD (Mercer); their Gram matrix is a similarity
matrix, and kernels are closed under +, ×, scaling.
The zoo: linear, polynomial (c + xTx′)d, and RBF exp(−γ∥x −x′∥2); tune γ, C by
cross-validation, standardize first.
The representer theorem gives f (x) = Pi αiK(xi, x); kernel ridge solves
α = (K + λI)−1y in closed form.
Cost: O(n2) Gram memory, ∼O(n3) to solve — superb for small/medium nonlinear
problems.
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 21 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 22

**Transcripción técnica de la diapositiva (extracción de texto):**

```text
References
C. M. Bishop (2006). Pattern Recognition and Machine Learning. Springer. (Ch. 6 kernel methods.)
T. Hastie, R. Tibshirani, J. Friedman (2009). The Elements of Statistical Learning, 2nd ed.
Springer. (Ch. 5 basis expansions & regularization; Ch. 12 support vector machines.)
K. P. Murphy (2022). Probabilistic Machine Learning: An Introduction. MIT Press. (Kernels &
kernel methods.)
Jan Hurtado (UTEC) Kernel Methods July 26, 2026 22 / 23
```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.


#### Página física 23

**Transcripción técnica de la diapositiva (extracción de texto):**

```text

```

**Visual:** en matrices Gram el brillo representa similitud alta y el oscuro baja según escala; los bloques diagonales son similitudes intraclase. Curvas RBF anchas/estrechas reflejan γ pequeño/grande.

## 7. Validaciones pendientes para convertirlo en especificación definitiva de P2

1. Contrastar contra la **Guía del Proyecto Final — Parte 2** para determinar algoritmos, entregables, métricas y restricciones exigidas.
2. Perfilar `support2.csv` para generar el diccionario de datos real: nombres, tipos, rangos, categorías, nulos y cardinalidad.
3. Revisar visualmente las páginas con diagramas de alta densidad, en particular las que contienen resultados numéricos en tablas o mapas de calor; el texto extraído puede fragmentar fórmulas, superíndices y símbolos.
4. Verificar que los criterios de elección de modelos, tratamiento de desbalance y umbrales estén alineados con el objetivo clínico y la rúbrica.

**Advertencia de fidelidad:** los apartados 1–5 son una sistematización técnica y propuesta de diseño; el apartado 6 conserva las diapositivas página por página y añade claves de lectura visual por tema, pero no pretende que cada píxel, anotación o gráfico haya sido descrito individualmente. Para un inventario visual exhaustivo por figura se requiere una segunda pasada de inspección de cada página renderizada.
