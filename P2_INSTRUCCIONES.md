# P2 — Instrucciones para entrenamiento de modelos

## 1. Dataset y particiones

Dataset: SUPPORT2
Variable objetivo: hospdead

- Entrenamiento: 7284 pacientes
- Prueba: 1821 pacientes
- Características seleccionadas: 27
- División estratificada: 80/20
- Semilla: 42

Las particiones se encuentran en:

- dataset/processed/train_indices.csv
- dataset/processed/test_indices.csv

## 2. Cargar los datos

Ejecutar desde la raíz del repositorio:

```python
from preprocessing import load_data, build_preprocessor

X_train, X_test, y_train, y_test = load_data()
```

## 3. Construir un modelo

Para regresión logística:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("preprocessor", build_preprocessor(scaled=True)),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
```

Para árboles y Random Forest:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

model = Pipeline([
    ("preprocessor", build_preprocessor(scaled=False)),
    ("classifier", RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)
```

## 4. Validación cruzada

Realizar la selección de modelos e hiperparámetros
exclusivamente sobre X_train e y_train.

Utilizar validación cruzada estratificada.

Mantener el preprocesador dentro del Pipeline,
para evitar fuga de información entre folds.

No utilizar X_test para seleccionar hiperparámetros.

## 5. Métricas

Reportar como mínimo:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Matriz de confusión

También se recomienda Average Precision,
por el desbalance entre las clases.

La clase positiva es hospdead = 1.

## 6. Baseline

Resultados de prueba del DummyClassifier(strategy="prior"):

- Accuracy: 0.7408
- Precision: 0.0000
- Recall: 0.0000
- F1-score: 0.0000
- ROC-AUC: 0.5000
- Average Precision: 0.2592

Archivo: results/tables/baseline_metrics.csv

## 7. Limitaciones metodológicas

El experimento utiliza características del dataset SUPPORT2
y se considera una clasificación retrospectiva
de mortalidad hospitalaria.

Las evaluaciones clínicas se relacionan con el tercer día
del estudio, que no necesariamente coincide con el
tercer día de hospitalización.

La información disponible no permite verificar las horas
exactas de las mediciones y de los desenlaces.

Se identificaron 138 pacientes fallecidos con
slos = 3 y d.time = 3.

Por ello, los resultados no deben interpretarse como
validación clínica prospectiva.

Consultar results/tables/temporal_audit.csv.
