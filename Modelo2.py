import matplotlib
matplotlib.use('TkAgg')
# Reimportar librerías necesarias
from keras.src.utils import pad_sequences
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.src.models import Sequential
from keras.src.layers import Masking, LSTM, Dense
from keras.src.optimizers import Adam
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Cargar el dataset
df = pd.read_csv('dataset_full.csv', sep=',')

# Reconstruir match_id y rally_uid
df['rally_diff'] = df['rally'].diff()
df['match_id'] = (df['rally_diff'] < 0).cumsum()
df.drop(columns='rally_diff', inplace=True)
df['match_id'] = df['match_id'].astype(str)
df['rally_uid'] = df['match_id'] + '_' + df['rally'].astype(str)

# Definir columnas para el modelo
# Se utilizan las columnas que se consideran relevantes para el modelo.
categorical_features = ['pass_rating', 'set_type', 'set_location', 'hit_type',
                        'block_touch', 'serve_type', 'win_reason',
                        'lose_reason', 'team']

numeric_features = ['receive_location', 'digger_location', 'pass_land_location',
                    'hitter_location', 'hit_land_location', 'num_blockers']

# Se definen las columnas de entrada y la columna objetivo
features = categorical_features + numeric_features + ['rally_uid']
# La columna objetivo es el equipo que gana el rally
target_col = 'winning_team'

# Copiar datos y limpiar
df_lstm = df[features + [target_col]].copy()

# Se imputan los valores nulos en las columnas categóricas y numéricas
# Las columnas categóricas se llenan con 'missing' y se convierten a string
for col in categorical_features:
    df_lstm[col] = df_lstm[col].fillna('missing').astype(str)
# Las columnas numéricas se llenan con -1
for col in numeric_features:
    df_lstm[col] = df_lstm[col].fillna(-1)

# Separar X (sin target)
X_input = df_lstm.drop(columns=[target_col])

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
X_prepared = preprocessor.fit_transform(X_input)
X_array = X_prepared.toarray()

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

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_array, test_size=0.2, random_state=42)

# Modelo LSTM
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_padded.shape[1], X_padded.shape[2])))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilación
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo LSTM
history = model.fit(X_train, y_train, epochs=130, batch_size=32, validation_data=(X_test, y_test))
model.save('lstm_model.h5')

# Evaluación
y_pred_probs = model.predict(X_test).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)

# Reporte
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No ganó', 'Ganó'], yticklabels=['No ganó', 'Ganó'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión LSTM')
plt.show()