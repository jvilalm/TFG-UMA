import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv('dataset_full.csv', sep=',')

# Columnas categóricas
categorical_features = ['pass_rating', 'set_type', 'set_location', 'hit_type',
                        'block_touch', 'serve_type', 'win_reason',
                        'lose_reason', 'winning_team', 'team']

# Columnas numéricas útiles para el modelo
features = categorical_features + [
    'receive_location', 'digger_location', 'pass_land_location',
    'hitter_location', 'hit_land_location', 'num_blockers'
]

# Imputar nulos
df_imputed = df[features].copy()
for col in categorical_features:
    df_imputed[col] = df_imputed[col].fillna('missing')
for col in ['receive_location', 'digger_location', 'pass_land_location',
            'hitter_location', 'hit_land_location', 'num_blockers']:
    df_imputed[col] = df_imputed[col].fillna(-1)

# Crear columna objetivo
# El modelo aprende a predecir si el equipo que está realizando esta jugada va a ser el que eventualmente gane el rally completo.

df_imputed['target'] = (df['team'] == df['winning_team']).astype(int)

# Separar variables
X = df_imputed.drop(columns='target')
y = df_imputed['target']

# Pipeline de preprocesamiento y modelo
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
# Pipeline de modelo
# Se utiliza un modelo de regresión logística para predecir la variable objetivo.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Entrenamiento del modelo
model.fit(X_train, y_train)
# Predicción de resultados
y_pred = model.predict(X_test)

# Clasificación del modelo
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No ganó', 'Ganó'])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión')
plt.grid(False)
plt.show()
