# Dataset

Este proyecto utiliza el dataset **SUPPORT**, específicamente el archivo `support2`.

El dataset forma parte del estudio **Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT)** y contiene información de pacientes hospitalizados con enfermedades graves.

## Fuente

La descripción y documentación del dataset se encuentran disponibles en:

https://hbiostat.org/data/repo/supportdesc

Antes de incluir el archivo directamente en un repositorio público, se recomienda revisar las condiciones de uso y distribución indicadas por la fuente.

## Archivo utilizado

El proyecto espera trabajar con el archivo:

```text
support2.csv
```

Una vez descargado, debe colocarse en:

```text
data/raw/support2.csv
```

La estructura esperada es:

```text
data/
├── README.md
└── raw/
    └── support2.csv
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

- `0` indica que el paciente sobrevivió al periodo hospitalario;
- `1` indica que el paciente falleció durante la hospitalización.

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
3. Crear la carpeta `data/raw/` si todavía no existe.
4. Copiar el archivo en:

```text
data/raw/support2.csv
```

5. Ejecutar los notebooks desde la carpeta `notebooks/`.

## Lectura desde los notebooks

Los notebooks deben cargar el archivo utilizando una ruta relativa al proyecto. Por ejemplo:

```python
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "support2.csv"

df_raw = pd.read_csv(DATA_PATH)
```

Esto evita depender de rutas locales específicas de una computadora.

## Datos originales

El archivo ubicado en `data/raw/` debe considerarse una copia de los datos originales.

No se recomienda modificarlo manualmente.

Cualquier transformación, limpieza, imputación o selección de variables debe realizarse mediante código dentro de los notebooks o scripts del proyecto.

Si posteriormente se decide guardar una versión procesada del dataset, puede utilizarse una estructura como:

```text
data/
├── raw/
│   └── support2.csv
└── processed/
    └── support2_processed.csv
```

La versión procesada debe poder regenerarse a partir del archivo original y del código del repositorio.

## Data leakage

Antes del modelado se identificaron variables que pueden incorporar información pronóstica previa o resultados de modelos ya desarrollados.

Las siguientes variables se consideran para exclusión como predictores principales:

```text
sps
aps
surv2m
surv6m
prg2m
prg6m
dnr
dnrday
```

La exclusión busca evitar que el desempeño de los modelos sea artificialmente optimista y mantener la comparación centrada en las variables demográficas, clínicas y fisiológicas originales.

## Reproducibilidad

Para reproducir el análisis es suficiente con:

1. obtener el archivo `support2.csv`;
2. colocarlo en `data/raw/`;
3. instalar las dependencias del proyecto;
4. ejecutar los notebooks en el orden indicado en el README principal.

No deben utilizarse rutas absolutas ni archivos locales que no estén documentados en el repositorio.
