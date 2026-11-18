# Predicción de mortalidad hospitalaria con Machine Learning

Proyecto académico desarrollado en la carrera de **Ciencia de la Computación** de la **Universidad de Ingeniería y Tecnología (UTEC)**.

## Integrantes

- Joaquin Geronimo Arriaga Cosio
- Marialejandra Bautista Gastelo
- Elizabeth Huaman Santillan

## Descripción del proyecto

El proyecto estudia la predicción de mortalidad hospitalaria en pacientes con enfermedades graves utilizando variables demográficas, clínicas y fisiológicas.

Se trabaja con el dataset **SUPPORT (Study to Understand Prognoses Preferences Outcomes and Risks of Treatment)**. La variable objetivo es `hospdead`, una variable binaria que indica si el paciente falleció durante la hospitalización.

La pregunta principal del trabajo es:

> ¿En qué medida las variables fisiológicas registradas durante la hospitalización permiten predecir la mortalidad hospitalaria con mayor capacidad discriminativa que las variables puramente demográficas?

El análisis parte de una etapa exploratoria para conocer la calidad y estructura de los datos antes de aplicar técnicas de preprocesamiento y modelado.

## Objetivo

Desarrollar y evaluar modelos de Machine Learning para predecir mortalidad hospitalaria y comparar el desempeño de un modelo basado únicamente en variables demográficas frente a modelos que incorporen información fisiológica y clínica.

## Dataset

Se utiliza el dataset `support2`, correspondiente al estudio SUPPORT.

Características principales:

- 9105 observaciones.
- 47 variables.
- Una observación por paciente.
- Variables demográficas, clínicas, fisiológicas y pronósticas.
- Variable objetivo: `hospdead`.

Entre las variables disponibles se encuentran edad, sexo, raza, diagnóstico principal, número de comorbilidades, presión arterial media, frecuencia cardíaca, frecuencia respiratoria, temperatura, razón PaO₂/FiO₂, albúmina, bilirrubina, creatinina, sodio, glucosa, BUN y recuento de leucocitos.

La variable objetivo se interpreta de la siguiente manera:

- `0`: el paciente sobrevivió al periodo hospitalario.
- `1`: el paciente falleció durante la hospitalización.

Las instrucciones para obtener y preparar el dataset se encuentran en [`dataset/README.md`](dataset/README.md).

## Análisis realizado

**P1 — Análisis exploratorio (EDA):**

- revisión de valores faltantes;
- análisis de la distribución de `hospdead`;
- comparación de variables fisiológicas entre pacientes sobrevivientes y fallecidos;
- análisis de correlación entre variables numéricas;
- identificación de valores atípicos mediante el rango intercuartílico (IQR);
- revisión de variables con riesgo de *data leakage*.

Para evitar resultados artificialmente optimistas se excluyen como predictores las puntuaciones de severidad, estimaciones de supervivencia, pronósticos médicos y órdenes de no reanimación (`sps`, `aps`, `surv2m`, `surv6m`, `prg2m`, `prg6m`, `dnr`, `dnrday`), así como las variables que solo se conocen después del desenlace o resumen toda la estadía (`death`, `slos`, `d.time`, `charges`, `totcst`, `totmcst`, `avtisst`, `sfdm2`, `adlsc`).

**P2 — Modelado y evaluación:**

- preprocesamiento dentro de `Pipeline` (sin fuga entre folds) y comparación de estrategias de imputación;
- baseline (`DummyClassifier`) y baseline demográfico;
- regresión logística, árbol de decisión, Random Forest y Gradient Boosting, ajustados con validación cruzada estratificada;
- métricas en prueba con intervalos bootstrap, calibración y recalibración;
- comparación demográficas vs. fisiológicas (pregunta de investigación);
- importancia de variables, análisis de errores y subgrupos;
- umbrales con costos clínicos explícitos y por grupo diagnóstico.

## Estructura del repositorio

```text
Prediccion_mortalidad_hospitalaria_ml/
│
├── README.md
├── LICENSE
├── requirements.txt
├── preprocessing.py               # Carga de particiones y preprocesador (sin ajustar)
│
├── src/                           # Código reutilizable (se importa)
│   ├── models.py                  # Pipelines y espacios de hiperparámetros
│   └── evaluation.py              # Métricas, umbrales, bootstrap y figuras
│
├── scripts/                       # Puntos de entrada (se ejecutan)
│   ├── run_p2.py                  # Experimento principal de P2
│   ├── run_p2_extensions.py       # Imputación, recalibración y umbrales por costo/grupo
│   └── make_paper_assets.py       # Tablas LaTeX y figuras del paper desde results/
│
├── tests/
│   └── test_pipeline.py           # Pruebas de integridad, imputación y no-fuga
│
├── dataset/
│   ├── README.md
│   ├── raw/support2.csv           # Datos originales (no se modifican)
│   └── processed/                 # Índices train/test y configuración
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_baseline.ipynb
│   └── 03_modeling_evaluation.ipynb
│
├── results/
│   ├── figures/                   # p2_*.png (principal), p2x_*.png (extensiones)
│   ├── tables/                    # baseline_*.csv, temporal_audit.csv, p2_*.csv, p2x_*.csv
│   ├── p2_run_metadata.json       # Semilla, versiones, modelo y umbral seleccionados
│   └── p2x_run_metadata.json
│
├── docs/
│   ├── Guia_Proyecto_Final_Machine_Learning_ES_ACTUALIZADO.pdf   # Guía oficial del proyecto
│   ├── P2_INSTRUCCIONES.md            # Contrato de datos P1 → P2
│   └── P2_metodologia_resultados.md   # Decisiones metodológicas y resultados de P2
│
└── paper/
    ├── Prediccion_Mortalidad_Hospitalaria_SUPPORT_P2.tex
    ├── Prediccion_Mortalidad_Hospitalaria_SUPPORT_P2.pdf   # Versión compilada (P2)
    ├── IEEEtran.cls
    ├── tables/                    # Generadas por scripts/make_paper_assets.py
    └── figures/                   # image*.png (P1) y p2*.png (copias de results/figures)
```

Los modelos serializados (`results/models/`) no se versionan: se regeneran con `scripts/run_p2.py`.

## Notebooks y scripts

Ejecutar en el siguiente orden:

1. `01_eda.ipynb` — Análisis exploratorio: valores faltantes, distribución de la variable objetivo, correlaciones, valores atípicos y riesgo de *data leakage*.
    - También disponible en Colab: [01_eda.ipynb](https://colab.research.google.com/drive/1jOFwgy5sXQwkk3EXLisNAm7pfX9ziHzh?usp=sharing)
2. `02_preprocessing_baseline.ipynb` — Exclusión de variables con fuga, partición estratificada 80/20, preprocesador, baseline `DummyClassifier` y auditoría temporal.
3. `python scripts/run_p2.py` — Selección de hiperparámetros por validación cruzada, evaluación en prueba, comparación demográficas vs. fisiológicas, interpretación y análisis de errores.
4. `python scripts/run_p2_extensions.py` — Comparación de imputaciones, recalibración de modelos con pesos de clase, umbral con costos clínicos explícitos y umbrales por grupo diagnóstico.
5. `03_modeling_evaluation.ipynb` — Presenta, explica y verifica los resultados de P2 (si faltan resultados, ejecuta los scripts automáticamente).
6. `python scripts/make_paper_assets.py` — Genera las tablas LaTeX (`paper/tables/`) y las figuras del paper a partir de `results/`.

La metodología y los resultados de P2 están documentados en [`docs/P2_metodologia_resultados.md`](docs/P2_metodologia_resultados.md).

## Requisitos

Se recomienda utilizar Python 3.10 o superior.

Las principales dependencias del proyecto son:

- pandas
- numpy
- matplotlib
- scikit-learn
- scipy, joblib
- nbformat, nbclient, ipykernel, pytest (ejecución de notebooks y pruebas)

Las versiones definitivas se encuentran en `requirements.txt`

## Instalación

Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd Prediccion_mortalidad_hospitalaria_ml
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en Windows:

```bash
.venv\Scripts\activate
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Preparación de los datos

Seguir las instrucciones indicadas en:

```text
dataset/README.md
```

El archivo final utilizado por los notebooks debe quedar en:

```text
dataset/raw/support2.csv
```

## Ejecución

Desde la raíz del repositorio:

```bash
python -m pytest -q             # pruebas de integridad, imputación y no-fuga
python scripts/run_p2.py                # experimento principal de P2 (~5 min)
python scripts/run_p2_extensions.py     # extensiones (~8 min)
python scripts/make_paper_assets.py     # tablas y figuras del paper
jupyter notebook                        # abrir los notebooks de notebooks/ en orden
```

Para compilar el paper se puede usar Overleaf o cualquier distribución LaTeX (pdflatex). Localmente se usó [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd paper
tectonic Prediccion_Mortalidad_Hospitalaria_SUPPORT_P2.tex
```

`python scripts/run_p2.py --quick` ejecuta una búsqueda reducida para pruebas rápidas y escribe en `results/quick/` (no versionado), sin sobrescribir los resultados oficiales.

Los notebooks deben poder ejecutarse desde un entorno limpio después de instalar las dependencias y preparar el dataset.

## Resultados

Las figuras y tablas del análisis se almacenan en:

```text
results/figures/
results/tables/
```

Todos los resultados incluidos en el paper se regeneran a partir del código del repositorio (`scripts/`), y las tablas del paper se construyen directamente desde `results/tables/`.

## Documento

El paper del proyecto se encuentra en la carpeta:

```text
paper/
```

El documento utiliza LaTeX con la clase `IEEEtran` en formato de conferencia.

## Estado del proyecto

- **P1** (EDA): completado.
- **P2** (modelado y evaluación): pipeline implementado y resultados generados. Se compararon regresión logística, árbol de decisión, Random Forest y Gradient Boosting frente al baseline. El modelo seleccionado por validación cruzada (Gradient Boosting) obtiene en prueba ROC-AUC 0.834 (IC 95 % 0.814–0.854), frente a 0.567 de un modelo solo con variables demográficas. Detalles en `docs/P2_metodologia_resultados.md`.
- Extensiones: la imputación por mediana se mantuvo (ninguna alternativa la mejora de forma relevante); la recalibración corrige la sobrestimación de riesgo de los modelos con pesos de clase; se analizaron umbrales con costos explícitos y por grupo diagnóstico.
- Paper actualizado con las secciones de P2 (`paper/Prediccion_Mortalidad_Hospitalaria_SUPPORT_P2.pdf`, 10 páginas).
- Pendiente: presentación de P2.

## Referencias

- Knaus, W. A., Harrell, F. E., Lynn, J., et al. *The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults*. Annals of Internal Medicine, 122:191–203, 1995.
- Vanderbilt University Department of Biostatistics. *SUPPORT Datasets*.  
  https://hbiostat.org/data/repo/supportdesc
