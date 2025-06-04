# Paso 1: Cargar y preparar el dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el dataset
df = pd.read_csv('dataset_full.csv', sep=',')

# Reconstruir match_id y rally_uid
df['rally_diff'] = df['rally'].diff()
df['match_id'] = (df['rally_diff'] < 0).cumsum().astype(str)
df.drop(columns='rally_diff', inplace=True)
df['rally_uid'] = df['match_id'] + '_' + df['rally'].astype(str)

# Definir columnas
categorical_features = ['pass_rating', 'set_type', 'set_location', 'block_touch',
                        'serve_type', 'win_reason', 'lose_reason', 'team', 'match_id']
numeric_features = ['receive_location', 'digger_location', 'pass_land_location',
                    'hitter_location', 'hit_land_location', 'num_blockers']

# Filtrar datos con hit_type no nulo
df_filtered = df[categorical_features + numeric_features + ['hit_type']].copy()
df_filtered = df_filtered[df_filtered['hit_type'].notna()]

# Imputar y formatear datos
for col in categorical_features:
    df_filtered[col] = df_filtered[col].fillna('missing').astype(str)
for col in numeric_features:
    df_filtered[col] = df_filtered[col].fillna(-1)

# X e y
X = df_filtered.drop(columns='hit_type')
y = df_filtered['hit_type'].astype(str)

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
X_encoded = preprocessor.fit_transform(X)

# Obtener nombres de columnas transformadas
ohe = preprocessor.named_transformers_['cat']
encoded_feature_names = list(ohe.get_feature_names_out(categorical_features)) + numeric_features

# Feature selection con RFE
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=20, step=10)
selector.fit(X_encoded, y)

# Seleccionar features
X_selected = X_encoded[:, selector.support_]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Escalar los datos y entrenar SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),  # Escalado necesario para SVM
    ('svc', SVC(kernel='rbf', C=1, gamma='scale'))  # Puedes ajustar kernel, C y gamma
])

# Entrenamiento
svm_pipeline.fit(X_train, y_train)

# Predicción
y_pred_svm = svm_pipeline.predict(X_test)

# Evaluación
print("Reporte de clasificación SVM:")
print(classification_report(y_test, y_pred_svm))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_svm, labels=svm_pipeline.classes_)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=svm_pipeline.classes_, yticklabels=svm_pipeline.classes_)
plt.title("Matriz de Confusión - SVM (hit_type)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
