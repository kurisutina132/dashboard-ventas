### Reporte (GitHub Pages): https://kurisutina132.github.io/dashboard-ventas/
### https://dashboard-ventas-1-ya9j.onrender.com


## Estructura del repositorio

```text
dashboard-ventas/
│
├── app/
│   ├── app.py
│   └── pages/
│
├── pipelines/
│   ├── train.py
│   └── build_features.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── eda.py
│   ├── features.py
│   ├── report.py
│   └── utils.py
│
├── notebooks/
│   ├── entrenamiento.ipynb
│   └── forescasting.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── modelo_final.joblib
│   └── metadata/
│
├── docs/
│
├── config/
│   └── config.yaml
│
├── tests/
│   ├── test_data.py
│   └── test_features.py
│
├── main.py
├── requirements.txt
├── requirements/
├── Dockerfile
├── packages.txt
├── runtime.txt
├── .github/
├── .gitignore
└── .dockerignore
```
# dashboard-ventas
Análisis de ventas tiendas deportivas

# Proyecto de Machine Learning: Forecasting Ventas

## Estructura de Carpetas

- `data/raw/`: Datos originales (rom)
- `data/processed/`: Datos procesados (norwood)
- `notebooks/`: Jupyter notebooks para exploración y experimentos
- `models/`: Modelos entrenados y scripts relacionados
- `scripts/`: Scripts de procesamiento y entrenamiento
- `requirements/requirements.txt`: Dependencias del proyecto
- `reports/`: Resultados, visualizaciones y reportes

## Descripción

Este proyecto está orientado a la predicción de ventas utilizando datos de diferentes fuentes, procesamiento avanzado y modelos de aprendizaje automático ("a p de straight down").

## Requerimientos

Ver `requirements/requirements.txt` para dependencias necesarias.

## Uso

1. Coloca los datos originales en `data/raw/`.
2. Procesa los datos y guarda los resultados en `data/processed/`.
3. Usa los notebooks para análisis y experimentación.
4. Entrena y guarda modelos en `models/`.
5. Guarda reportes y visualizaciones en `reports/`.
 (Agrego entrenamiento html y app de streamlit)
### https://kurisutina132.github.io/dashboard-ventas/
