# Predicción de mortalidad hospitalaria con Machine Learning

Proyecto académico desarrollado en la carrera de **Ciencia de la Computación** de la **Universidad de Ingeniería y Tecnología (UTEC)**.

## Integrantes

- Joaquin Geronimo Arriaga Cosio
- Marialejandra Bautista Gastelo
- Elizabeth Huaman Santillan

Lima, Perú.

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

Las instrucciones para obtener y preparar el dataset se encuentran en [`data/README.md`](data/README.md).

## Análisis realizado

La primera etapa del proyecto corresponde al análisis exploratorio de datos (EDA). Actualmente se incluyen los siguientes puntos:

- revisión de valores faltantes;
- análisis de la distribución de `hospdead`;
- comparación de variables fisiológicas entre pacientes sobrevivientes y fallecidos;
- análisis de correlación entre variables numéricas;
- identificación de valores atípicos mediante el rango intercuartílico (IQR);
- revisión de variables con riesgo de *data leakage*.

Para reducir el riesgo de obtener resultados artificialmente optimistas, se consideran para exclusión como predictores principales las variables:

`SPS`, `APS`, `surv2m`, `surv6m`, `prg2m`, `prg6m`, `dnr` y `dnrday`.

Estas variables contienen puntuaciones de severidad, estimaciones de supervivencia, pronósticos médicos o información relacionada con órdenes de no reanimación.

## Estructura del repositorio

```text
support-mortality-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   └── raw/
│       └── support2.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   └── utils.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── paper/
│   ├── main.tex
│   ├── IEEEtran.cls
│   └── figures/
│
└── docs/
    └── project_notes.md
```
## Notebooks

Los notebooks se ejecutarán en el siguiente orden:

1. `01_eda.ipynb`  
   Análisis exploratorio, valores faltantes, distribución de la variable objetivo, correlaciones, valores atípicos y revisión de *data leakage*.

2. `02_preprocessing.ipynb`  
   Preparación de los datos, imputación, codificación, escalamiento y selección de variables.

3. `03_baseline.ipynb`  
   Construcción y evaluación de un modelo base utilizando principalmente variables demográficas.

4. `04_modeling.ipynb`  
   Entrenamiento de los modelos seleccionados incorporando variables clínicas y fisiológicas.

5. `05_evaluation.ipynb`  
   Comparación de resultados y evaluación final mediante las métricas definidas para el proyecto.

## Requisitos

Se recomienda utilizar Python 3.10 o superior.

Las principales dependencias del proyecto son:

- pandas
- numpy
- matplotlib
- scikit-learn

Las versiones definitivas se encuentran en `requirements.txt`

## Instalación

Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd support-mortality-ml
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
data/README.md
```

El archivo final utilizado por los notebooks debe quedar en:

```text
data/raw/support2.csv
```

## Ejecución

Iniciar Jupyter:

```bash
jupyter notebook
```

Luego abrir los notebooks desde la carpeta `notebooks/` y ejecutarlos en orden.

Los notebooks deben poder ejecutarse desde un entorno limpio después de instalar las dependencias y preparar el dataset.

## Resultados

Las figuras y tablas relevantes para el análisis se almacenarán en:

```text
results/figures/
results/tables/
```

El objetivo es que los resultados incluidos en el paper puedan ser generados nuevamente a partir del código del repositorio.

## Documento

El paper del proyecto se encuentra en la carpeta:

```text
paper/
```

El documento utiliza LaTeX con la clase `IEEEtran` en formato de conferencia.

## Estado del proyecto

Actualmente se encuentra desarrollada la etapa de análisis exploratorio. Las siguientes etapas corresponden al preprocesamiento, construcción del baseline, entrenamiento de modelos y evaluación final.

## Referencias

- Knaus, W. A., Harrell, F. E., Lynn, J., et al. *The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults*. Annals of Internal Medicine, 122:191–203, 1995.
- Vanderbilt University Department of Biostatistics. *SUPPORT Datasets*.  
  https://hbiostat.org/data/repo/supportdesc
