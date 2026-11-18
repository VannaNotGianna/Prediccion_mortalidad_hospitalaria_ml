# Dataset

Este proyecto utiliza el dataset **SUPPORT**, específicamente el archivo `support2`.

El dataset forma parte del estudio **Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT)** y contiene información de pacientes hospitalizados con enfermedades graves.

## Fuente

La descripción y documentación del dataset se encuentran disponibles en:

https://hbiostat.org/data/repo/supportdesc

## Archivo utilizado

El proyecto espera trabajar con el archivo:

```text
support2.csv
```

Una vez descargado, debe colocarse en:

```text
dataset/raw/support2.csv
```

La estructura esperada es:

```text
dataset/
├── README.md
├── raw/
│   └── support2.csv                  # Datos originales (no se modifican)
└── processed/
    ├── train_indices.csv             # Índices originales del conjunto de entrenamiento (7284)
    ├── test_indices.csv              # Índices originales del conjunto de prueba (1821)
    └── preprocessing_config.json     # Variables seleccionadas, excluidas y parámetros
```

## Descripción general

El conjunto utilizado en el proyecto contiene:

- 9105 observaciones;
- 47 variables;
- una fila por paciente;
- variables demográficas, clínicas, fisiológicas y pronósticas.

La variable objetivo es:

```text
hospdead
```

donde:

- `0` indica que el paciente sobrevivió al periodo hospitalario (6745 pacientes, 74.08 %);
- `1` indica que el paciente falleció durante la hospitalización (2360 pacientes, 25.92 %).

Entre las variables disponibles se encuentran:

- edad;
- sexo;
- raza;
- diagnóstico principal;
- número de comorbilidades;
- presión arterial media;
- frecuencia cardíaca;
- frecuencia respiratoria;
- temperatura;
- razón PaO₂/FiO₂;
- albúmina;
- bilirrubina;
- creatinina;
- sodio;
- glucosa;
- BUN;
- recuento de leucocitos.

## Preparación

1. Descargar el dataset desde la fuente indicada.
2. Verificar que el archivo tenga el nombre `support2.csv`.
3. Crear la carpeta `dataset/raw/` si todavía no existe.
4. Copiar el archivo en:

```text
dataset/raw/support2.csv
```

5. Ejecutar los notebooks desde la carpeta `notebooks/` o los scripts desde la raíz del repositorio (ver el README principal).

## Lectura desde el código

La forma recomendada de cargar los datos es mediante el módulo `preprocessing.py` de la raíz, que recupera exactamente las particiones fijadas en P1:

```python
from preprocessing import load_data

X_train, X_test, y_train, y_test = load_data()
```

Si se necesita leer el archivo original directamente, usar una ruta relativa al proyecto:

```python
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = PROJECT_ROOT / "dataset" / "raw" / "support2.csv"

df_raw = pd.read_csv(DATA_PATH)
```

Esto evita depender de rutas locales específicas de una computadora.

## Datos originales y procesados

El archivo ubicado en `dataset/raw/` es una copia de los datos originales y no debe modificarse.

Cualquier transformación, limpieza, imputación o selección de variables se realiza mediante código. Los datos procesados **no** se guardan como un nuevo CSV: en `dataset/processed/` solo se guardan los índices de la partición train/test y la configuración. La imputación, codificación y escalamiento se aprenden dentro de un `Pipeline`, ajustado únicamente con los datos de entrenamiento, para evitar fuga de información.

Los archivos de `dataset/processed/` se generan con `notebooks/02_preprocessing_baseline.ipynb` y pueden regenerarse a partir del archivo original (partición estratificada 80/20, semilla 42).

## Data leakage

Antes del modelado se identificaron variables que no deben usarse como predictores. Todas se registran en `preprocessing_config.json` (`initial_excluded_columns`):

- **Puntuaciones y pronósticos previos:** `sps`, `aps`, `surv2m`, `surv6m`, `prg2m`, `prg6m`.
- **Órdenes de no reanimación:** `dnr`, `dnrday`.
- **Desenlaces o información posterior / de toda la estadía:** `death`, `slos`, `d.time`, `charges`, `totcst`, `totmcst`, `avtisst`, `sfdm2`, `adlsc`.

Además, se excluyen `adlp` y `urine` por superar el 50 % de valores faltantes en entrenamiento.

La exclusión busca evitar que el desempeño de los modelos sea artificialmente optimista y mantener la comparación centrada en las variables demográficas, clínicas y fisiológicas originales.

## Reproducibilidad

Para reproducir el análisis es suficiente con:

1. obtener el archivo `support2.csv`;
2. colocarlo en `dataset/raw/`;
3. instalar las dependencias del proyecto;
4. ejecutar los notebooks y scripts en el orden indicado en el README principal.

No deben utilizarse rutas absolutas ni archivos locales que no estén documentados en el repositorio.
