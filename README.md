![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-red)
![pandas](https://img.shields.io/badge/pandas-3.0.0-150458?logo=pandas)
![Status](https://img.shields.io/badge/Estado-Completado-brightgreen)

# Predicción de Pobreza en Costa Rica — Consultoría BID (2025)

Proyecto de clasificación binaria desarrollado para el **Banco Interamericano de Desarrollo (BID)** con el objetivo de mejorar el sistema de focalización de subsidios sociales en Costa Rica, reemplazando el *Proxy Means Test* (PMT) vigente. Autores: David Rodríguez y Juan Rueda.

---

## Problema

El gobierno de Costa Rica clasifica hogares vulnerables usando un PMT desactualizado que genera exclusiones injustas (falsos negativos) e inclusiones innecesarias (falsos positivos). El BID contrató este equipo para construir un modelo de clasificación que, a partir de características observables del hogar, prediga con mayor precisión si un hogar es **pobre** (categorías 1–2 del sistema SINIRUBE) o **no pobre** (categorías 3–4).

La asimetría de costos es clave: excluir un hogar pobre cuesta **USD $3,880** (pérdida de bienestar, emergencias sociales, costo reputacional), mientras que incluir uno no pobre cuesta **USD $1,350** (subsidio + administrativo). Esta razón de 2.87x justifica priorizar el *recall* sobre la precisión.

---

## Datos

- **Fuente:** [Costa Rican Household Poverty Level Prediction — Kaggle/BID](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/data) (sistema SINIRUBE)
- **Nivel original:** individual — agregado a nivel hogar por regla de mayoría
- **Variables:** 143 originales → 50 tras limpieza y eliminación de multicolinealidad (VIF)
- **Target binarizado:** 22.9% de hogares clasificados como pobres
- **Partición:** 80/20 estratificado con semilla fija para reproducibilidad

---

## Metodología

### 1. Exploración y preprocesamiento
Análisis exploratorio, tratamiento de valores faltantes, eliminación de redundancias e inconsistencias. Agregación individual a nivel hogar documentada con justificación por variable.

### 2. Feature Engineering (5 variables nuevas)
| Variable | Fundamento económico |
|---|---|
| Índice de Activos Tech | Proxy de ingreso permanente (Filmer & Pritchett) |
| Privación de Servicios Básicos | Pobreza multidimensional (metodología OPHI) |
| Calidad de Materiales de Vivienda | Indicador de pobreza estructural crónica |
| Tasa de Dependencia del Hogar | Carga económica y restricción de movilidad social |
| Hacinamiento Severo | Determinante intergeneracional de bienestar |

### 3. Métrica de optimización
**F₂-score** (β = 2): pondera el *recall* el doble que la *precision*, coherente con la asimetría de costos del programa.

### 4. Modelos entrenados
Tres modelos con tuning vía `GridSearchCV` / `RandomizedSearchCV` (5-fold estratificado, optimizando F₂):
- Regresión Logística (regularización L2)
- Random Forest
- XGBoost

### 5. Interpretabilidad
Análisis de importancia global de variables + explicación individual hogar por hogar usando coeficientes del modelo logístico, permitiendo responder apelaciones ciudadanas concretas.

---

## Hallazgos principales

| Modelo | Umbral | Accuracy | Precision | Recall | F₁ | **F₂** | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Regresión Logística** (C=0.01, L2) | **0.35** | 57.9% | 34.1% | **89.3%** | 49.3% | **67.5%** | 77.5% |
| Random Forest (700 árboles) | 0.40 | 53.9% | 32.1% | 90.3% | 47.3% | 66.2% | 77.2% |
| XGBoost (600 est., max_depth=3) | 0.15 | 67.0% | 37.3% | 64.1% | 47.1% | 56.0% | 74.9% |

**Modelo seleccionado:** Regresión Logística con umbral 0.35 — mayor F₂ (67.5%) y Recall (89.3%), superando al PMT vigente.

**Variables más determinantes:** Índice de Activos Tech (+), Promedio de Años de Escolaridad (+), Cantidad de Niños (−), Calidad de Materiales de Vivienda (−), Presencia de Persona con Discapacidad (−).

**Perfil geográfico:** Brunca (35.3%) y Pacífico Central (33.3%) lideran la tasa de pobreza; zonas rurales superan en 7 puntos porcentuales a las urbanas (28% vs. 21%).

---

## Estructura del repositorio

```
├── Datos/
│   ├── train.csv                   # Datos originales (nivel individuo)
│   ├── train_cleaned_hogar.csv     # Datos agregados a nivel hogar
│   └── codebook.csv                # Diccionario de variables
├── Modelos/
│   ├── Punto1_modelo.pkl           # Modelo logístico serializado
│   └── scaler.pkl                  # Escalador entrenado
├── Visualizaciones/                # Gráficos del análisis exploratorio
├── Punto1_Clasificacion.ipynb      # Notebook principal (pipeline completo)
├── Punto1_Informe_BID.pdf          # Informe ejecutivo para el BID
├── Enunciado.pdf                   # Especificaciones del proyecto
└── requirements.txt
```
