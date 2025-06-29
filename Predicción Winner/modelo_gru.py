# -*- coding: utf-8 -*-
"""Modelo_GRU.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lC5Apm8s8WgoD1EeBQ3JxOyN8kCt9rHz

# Modelo GRU
"""

from keras.src.utils import pad_sequences
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.src.models import Sequential
from keras.src.layers import Masking, GRU, Dense, LeakyReLU, Dropout
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import brier_score_loss

# Cargar el dataset
df = pd.read_csv('../dataset_full.csv', sep=',')

"""## Selección de Columnas Categóricas
Se realizará una selección de columnas categóricas sobre las que aplicaremos ***one hot encoding*** ya que son columnas que contienen variables categóricas (tipo texto o etiquetas). Dentro de estas columnas encontramos: `pass_rating`, `set_type`, `set_location`, `hit_type`, `block_touch`, `serve_type`, `win_reason`, `lose_reason`, `winning_team` y `team`.
Añadiremos el resto de columnas útiles para entrenar el modelo dentro de la variable `features`.

Se han añadido las columnas `match_id` y `rally_uid` que después nos permitirán realizar la división de las diferentes secuencias.
"""

# Reconstruir match_id y rally_uid
df['rally_diff'] = df['rally'].diff()
df['match_id'] = (df['rally_diff'] < 0).cumsum()
df.drop(columns='rally_diff', inplace=True)
df['match_id'] = df['match_id'].astype(str)
df['rally_uid'] = df['match_id'] + '_' + df['rally'].astype(str)

# Definir columnas para el modelo
# Se utilizan las columnas que se consideran relevantes para el modelo.
categorical_features = ['pass_rating', 'set_type', 'set_location', 'hit_type',
                        'block_touch', 'serve_type', 'team']

numeric_features = ['receive_location', 'digger_location', 'pass_land_location',
                    'hitter_location', 'hit_land_location', 'num_blockers']

"""## Target
En este primer modelo crearemos una columna target con el objetivo de predecir si el equipo que está realizando la ronda será el ganador del rally.

"""

# Se definen las columnas de entrada y la columna objetivo
features = categorical_features + numeric_features
# La columna objetivo es el equipo que gana el rally
target_col = 'winning_team'

"""## Imputar Nulos
Al entrenar el modelo tenemos que saber que las filas que contengan algún valor nulo no nos servirán para entrenar el mismo. Es por ello que se han modificado las filas con valores nulos, sustityuendo estos por el valor `mising` en las columnas categóricas y por el valor `-1` en las columnas numéricas.
"""

# Dataset para procesamiento (mantenemos rally_uid solo para agrupación)
df_lstm = df[categorical_features + numeric_features + ['rally_uid', target_col]].copy()

# Se imputan los valores nulos en las columnas categóricas y numéricas
# Las columnas categóricas se llenan con 'missing' y se convierten a string
for col in categorical_features:
    df_lstm[col] = df_lstm[col].fillna('missing').astype(str)

# Las columnas numéricas se llenan con -1
for col in numeric_features:
    df_lstm[col] = df_lstm[col].fillna(-1)

"""## Procesamiento
Separaremos la columna target del resto. Aplicaremos finalmente one hot encoder sobre las columnas categóricas y crearemos el modelo de regresión logística
"""

# Separar X (sin target)
X_input = df_lstm.drop(columns=[target_col, 'rally_uid'])

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
X_prepared = preprocessor.fit_transform(X_input)
X_array = X_prepared

# Crear target numérico
df_lstm['target'] = (df_lstm[target_col] == 'b').astype(int)

# Agrupar jugadas por rally (secuencias)
df_lstm['rally_index'] = df_lstm.groupby('rally_uid').cumcount()
df_lstm['sequence_id'] = df_lstm['rally_uid']

rally_sequences = []
rally_labels = []

# Agrupar por rally_uid y crear secuencias
for rally_id, group in df_lstm.groupby('rally_uid'):
    rally_seq = X_array[group.index]
    label = group['target'].iloc[0]
    rally_sequences.append(rally_seq)
    rally_labels.append(label)

# Ajustar longitud de secuencias
X_padded = pad_sequences(rally_sequences, padding='post', dtype='float32')
y_array = np.array(rally_labels)

"""## Entrenamiento y clasificación del modelo

Finalmente dividiremos los datos de forma aleatoria en un 80% para entrenar el modelo y un 20% sobre el que aplicaremos el mismo para validarlo.
"""

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_array, test_size=0.2, random_state=42)

# Calcular pesos para balancear clases
weights = compute_class_weight(class_weight='balanced',
                                classes=np.unique(y_train),
                                y=y_train)
class_weights = dict(zip(np.unique(y_train), weights))

# Definir el modelo
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_padded.shape[1], X_padded.shape[2])))
model.add(GRU(64))
model.add(Dropout(0.3))  # Regularización
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(1, activation='sigmoid'))

# Compilación
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callback para detener si no mejora
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Entrenamiento
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights,
                    callbacks=[early_stop])

# Convertir el historial a DataFrame
history_df = pd.DataFrame(history.history)

# Guardar a CSV
history_df.to_csv("training_history_gru.csv", index=False)

print("Historial de entrenamiento guardado como 'training_history.csv'")

# Evaluación
y_pred_probs = model.predict(X_test).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)

"""## Resultados"""

# Mostrar la curva ROC usando las probabilidades predichas
RocCurveDisplay.from_predictions(y_test, y_pred_probs)
plt.title('Curva ROC (GRU)')
plt.show()

# Calcular AUC
auc = roc_auc_score(y_test, y_pred_probs)
print(f'AUC: {auc:.4f}')

# Calcular MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Error Absoluto Medio (MAE): {mae:.4f}')

# Calcular Brier Score
brier = brier_score_loss(y_test, y_pred_probs)
print(f'Brier Score: {brier:.4f}')
# Reporte
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No ganó', 'Ganó'], yticklabels=['No ganó', 'Ganó'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión GRU')
plt.show()