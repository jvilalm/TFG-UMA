# Paso 1: Cargar y preparar el dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('dataset_full.csv', sep=',')

# Reconstruir match_id y rally_uid
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Reconstruir match_id y rally_uid
df['rally_diff'] = df['rally'].diff()
df['match_id'] = (df['rally_diff'] < 0).cumsum().astype(str)
df.drop(columns='rally_diff', inplace=True)
df['rally_uid'] = df['match_id'] + '_' + df['rally'].astype(str)

# Definir columnas
categorical_features = ['pass_rating', 'set_type', 'set_location', 'block_touch',
                        'serve_type', 'win_reason', 'lose_reason', 'team', 'match_id']
numeric_features = ['receive_location', 'digger_location', 'pass_land_location',
                    'hitter_location', 'num_blockers', 'hit_land_location']

# Filtrar datos con hit_land_location no nulo
df_filtered = df[categorical_features + numeric_features].copy()
df_filtered = df_filtered[df_filtered['hit_land_location'].notna()]

# Separar X e y
y = df_filtered['hit_land_location'].astype(str)
X = df_filtered.drop(columns='hit_land_location')

# Imputar y formatear datos
for col in categorical_features:
    X[col] = X[col].fillna('missing').astype(str)
for col in numeric_features:
    X[col] = X[col].fillna(-1)

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
encoded_feature_names = list(ohe.get_feature_names_out(categorical_features)) + \
                        [col for col in numeric_features if col != 'hit_land_location']

# Selección de características con RFE
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=20, step=10)
selector.fit(X_encoded, y)

# Seleccionar features
X_selected = X_encoded[:, selector.support_]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluar modelo
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

