# TFG - Inteligencia Artificial Aplicada al Voleibol: Modelos de Predicción para Apoyar la Toma de Decisiones

Este repositorio contiene el código de un Trabajo de Fin de Grado (TFG) orientado al análisis del dataset VREN y a la generación de modelos predictivos para acciones de voleibol.

## Contenido del repositorio

- `dataset_full.csv`: conjunto de datos con 2.429 registros y 18 columnas.
- `EDA.py`: análisis exploratorio de los datos.
- `Predicción Winner/`: scripts de modelos (regresión logística, RNN, GRU y LSTM) para predecir el equipo ganador de cada rally.
- `Predicción Hit/`: script para predecir el tipo de remate.
- `images/`: gráficos generados durante el análisis y el entrenamiento de modelos.

## Columnas del dataset

```
rally, round, team, receive_location, digger_location, pass_land_location,
hitter_location, hit_land_location, pass_rating, set_type, set_location,
hit_type, num_blockers, block_touch, serve_type, win_reason, lose_reason,
winning_team
```

## Ejecución

Todos los scripts se han desarrollado en Python. Para ejecutar cualquiera de ellos se recomienda tener instaladas librerías como `pandas`, `numpy`, `scikit-learn`, `keras` y `matplotlib`.

Ejemplo para lanzar el EDA:

```bash
python EDA.py
```

Los modelos de la carpeta **Predicción Winner** y **Predicción Hit** pueden ejecutarse de la misma manera, asegurándose de que el archivo `dataset_full.csv` está disponible en el directorio raiz o en la ruta relativa correspondiente.

---

Se incluyen diversos gráficos y resultados de entrenamiento en la carpeta `images/` y en los archivos `training_history_*.csv`.
