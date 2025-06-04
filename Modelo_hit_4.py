import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from keras.src.models import Sequential
from keras.src.layers import GRU, Dense, Dropout, Masking, LeakyReLU
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping
from keras.src.utils import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Cargar y preparar el dataset ===
df = pd.read_csv('dataset_full.csv', sep=',')
df['rally_diff'] = df['rally'].diff()
df['match_id'] = (df['rally_diff'] < 0).cumsum().astype(str)
df.drop(columns='rally_diff', inplace=True)
df['rally_uid'] = df['match_id'] + '_' + df['rally'].astype(str)

categorical_features = ['pass_rating', 'set_type', 'set_location', 'block_touch',
                        'serve_type', 'win_reason', 'lose_reason', 'team', 'match_id']
numeric_features = ['receive_location', 'digger_location', 'pass_land_location',
                    'hitter_location', 'hit_land_location', 'num_blockers']
target_col = 'hit_type'
context_size = 1

df_filtered = df[categorical_features + numeric_features + ['rally_uid', target_col]].copy()
df_filtered = df_filtered[df_filtered[target_col].notna()]

for col in categorical_features:
    df_filtered[col] = df_filtered[col].fillna('missing').astype(str)
for col in numeric_features:
    df_filtered[col] = df_filtered[col].fillna(-1)

# === 2. Codificar el target ===
label_encoder = LabelEncoder()
df_filtered['target'] = label_encoder.fit_transform(df_filtered[target_col].astype(str))
n_classes = len(label_encoder.classes_)

# === 3. Preprocesamiento de features ===
X = df_filtered.drop(columns=[target_col, 'target'])
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')
X_encoded = preprocessor.fit_transform(X.drop(columns='rally_uid'))
X_array = X_encoded.toarray()

# === 4. Generar muestras por jugada ===
df_filtered = df_filtered.reset_index(drop=True)
df_filtered['row_index'] = df_filtered.index
sequence_inputs, sequence_targets = [], []

for rally_id, group in df_filtered.groupby('rally_uid'):
    group = group.sort_index()
    row_indices = group['row_index'].values
    targets = group['target'].values
    for i in range(1, len(group)):
        start = max(0, i - context_size)
        indices = row_indices[start:i]
        sequence = X_array[indices]
        sequence_inputs.append(sequence)
        sequence_targets.append(targets[i])

X_padded = pad_sequences(sequence_inputs, padding='pre', dtype='float32')
y_array = np.array(sequence_targets)

# === 5. División de datos ===
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_array, test_size=0.2, random_state=42)

# === 6. Calcular class weights ===
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), weights))

# === 7. Modelo GRU ===
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(X_padded.shape[1], X_padded.shape[2])))
model.add(GRU(64))
model.add(Dropout(0.3))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.01))
model.add(Dense(n_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# === 8. Entrenar ===
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    batch_size=32,
                    class_weight=class_weights,
                    callbacks=[early_stop])

# === 9. Evaluación ===
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

labels_presentes = np.unique(y_test)
target_names = [label_encoder.classes_[i] for i in labels_presentes]

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes, labels=labels_presentes)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - GRU (hit_type por jugada)')
plt.tight_layout()
plt.show()

# Reporte
print(classification_report(y_test, y_pred_classes,
                            labels=labels_presentes,
                            target_names=target_names))
