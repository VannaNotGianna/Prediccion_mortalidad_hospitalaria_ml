# Modelo predictivo de Mortalidad Hospitalaria en pacientes con enfermedades graves

Proyecto desarrollado como parte de la carrera de **Ciencia de la Computación** en la **Universidad de Ingeniería y Tecnología (UTEC)**.

## Integrantes

- Joaquin Geronimo Arriaga Cosio
- Marialejandra Bautista Gastelo
- Elizabeth Huaman Santillan

## Descripción

Este proyecto aborda la predicción de mortalidad hospitalaria en pacientes con enfermedades graves utilizando información demográfica, clínica y fisiológica. El trabajo se basa en el dataset **SUPPORT (Study to Understand Prognoses Preferences Outcomes and Risks of Treatment)** y plantea un problema de clasificación binaria cuya variable objetivo es `hospdead`.

La pregunta principal del proyecto es si las variables fisiológicas registradas durante la hospitalización permiten discriminar mejor el riesgo de mortalidad que un conjunto de variables puramente demográficas.

## Abstract

En este proyecto se propone desarrollar un modelo de Machine Learning para predecir la mortalidad hospitalaria en pacientes con enfermedades graves utilizando variables demográficas, clínicas y fisiológicas. Para el análisis se utilizó el dataset SUPPORT, que contiene 9105 observaciones y 47 variables, considerando como variable objetivo `hospdead`, donde 1 representa el fallecimiento del paciente durante la hospitalización y 0 indica que el paciente sobrevivió. Como parte del análisis exploratorio se evaluaron los valores faltantes, la distribución de la variable objetivo, las diferencias en variables fisiológicas entre pacientes sobrevivientes y fallecidos, la correlación entre variables numéricas y la presencia de valores atípicos. También se identificaron variables que podrían generar data leakage, como puntuaciones de severidad y estimaciones pronósticas previas, por lo que no fueron consideradas como predictores principales. Este análisis permite establecer una base para las siguientes etapas de preprocesamiento y modelado, donde se evaluará si las variables fisiológicas permiten obtener una mejor capacidad predictiva que un modelo basado únicamente en variables demográficas.

## Dataset

Se utiliza el dataset `support2`, correspondiente al estudio SUPPORT. El conjunto contiene:

- **9105 observaciones**
- **47 variables**
- Una observación por paciente
- Variables demográficas, clínicas, fisiológicas y pronósticas
- Variable objetivo: `hospdead`

Entre las variables analizadas se encuentran edad, sexo, raza, diagnóstico principal, número de comorbilidades, presión arterial media, frecuencia cardíaca, frecuencia respiratoria, temperatura, razón PaO₂/FiO₂, albúmina, bilirrubina, creatinina, sodio, glucosa, BUN y recuento de leucocitos.

La variable `hospdead` se interpreta de la siguiente manera:

- `0`: el paciente sobrevivió al periodo hospitalario.
- `1`: el paciente falleció durante la hospitalización.

## Análisis exploratorio

La primera etapa del proyecto se concentra en conocer la estructura y calidad de los datos antes de realizar el modelado. El EDA incluye:

- revisión de valores faltantes;
- análisis de la distribución de `hospdead`;
- comparación de variables fisiológicas entre pacientes sobrevivientes y fallecidos;
- análisis de correlación de Pearson entre variables numéricas y mortalidad hospitalaria;
- identificación de valores atípicos mediante el rango intercuartílico (IQR).

Los valores extremos no se eliminan de forma automática, ya que en una población de pacientes con enfermedades graves algunas mediciones fisiológicas extremas pueden corresponder a condiciones clínicas reales.

## Prevención de data leakage

Antes del modelado se identificaron variables que pueden incorporar información pronóstica previa o resultados de modelos desarrollados anteriormente. Para evitar estimaciones artificialmente optimistas, se consideran para exclusión como predictores principales:

` sps `, ` aps `, ` surv2m `, ` surv6m `, ` prg2m `, ` prg6m `, ` dnr ` y ` dnrday `.

Esta decisión busca que la comparación se concentre en la capacidad predictiva de las características demográficas, clínicas y fisiológicas originales.

## Hipótesis

Se espera que los modelos que incorporen variables fisiológicas tengan una mayor capacidad discriminativa que un baseline basado únicamente en variables demográficas, debido a que estas mediciones reflejan de forma más directa el estado clínico del paciente.

## Etapas del proyecto

El trabajo se encuentra organizado en las siguientes etapas:

1. Introducción y formulación del problema.
2. Datos y análisis exploratorio de datos.
3. Preprocesamiento e ingeniería de características.
4. Metodología y modelos.
5. Diseño experimental y evaluación.
6. Resultados y análisis.
7. Discusión y limitaciones.
8. Conclusiones.

Actualmente, el análisis exploratorio establece la base para definir posteriormente las estrategias de imputación, codificación, escalamiento y selección de variables.

## Documento

El paper está preparado en **LaTeX** utilizando la clase `IEEEtran` en formato de conferencia IEEE.

La estructura principal esperada del proyecto es:

```text
.
├── support_project_ieee.tex
├── IEEEtran.cls
├── README.md
└── figures/
    ├── image1.png
    ├── image2.png
    ├── ...
    └── image8.png
```

Para compilar el documento, `IEEEtran.cls` debe mantenerse en el mismo directorio que el archivo `.tex`, o estar disponible en la distribución de LaTeX utilizada.

## Referencias

1. W. A. Knaus, F. E. Harrell, J. Lynn, et al. *The SUPPORT prognostic model: Objective estimates of survival for seriously ill hospitalized adults*. Annals of Internal Medicine, 122:191–203, 1995.
2. Vanderbilt University Department of Biostatistics. *SUPPORT Datasets*: https://hbiostat.org/data/repo/supportdesc

