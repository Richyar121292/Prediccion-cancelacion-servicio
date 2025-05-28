import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve

# ==============================
# 1. Carga y limpieza de datos
# ==============================
def load_data(path):
    """Carga los datasets desde la ruta especificada."""
    contract = pd.read_csv(path + 'contract.csv')
    personal = pd.read_csv(path + 'personal.csv')
    internet = pd.read_csv(path + 'internet.csv')
    phone = pd.read_csv(path + 'phone.csv')

    return contract, personal, internet, phone

def clean_data(contract, personal, internet, phone):
    """Une y limpia los datasets."""
    merged_data = contract.merge(personal, on='CustomerID', how='left')\
                          .merge(internet, on='CustomerID', how='left')\
                          .merge(phone, on='CustomerID', how='left')

    # Renombrar columnas
    merged_data.rename(columns={'CustomerID': 'CustomerId', 'gender': 'Gender'}, inplace=True)

    # Convertir fechas a datetime y manejar valores ausentes
    merged_data['BeginDate'] = pd.to_datetime(merged_data['BeginDate'], format='%Y-%m-%d', errors='coerce')
    merged_data['EndDate'] = merged_data['EndDate'].replace('No', pd.NaT)
    merged_data['EndDate'] = pd.to_datetime(merged_data['EndDate'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Crear columna 'OriginalChurn'
    merged_data['OriginalChurn'] = merged_data['EndDate'].isna().astype(int)

    # Calcular duración del contrato
    merged_data['ContractDuration'] = (merged_data['EndDate'] - merged_data['BeginDate']).dt.days
    median_duration = merged_data['ContractDuration'].dropna().median()
    merged_data['EndDate'] = merged_data['EndDate'].fillna(merged_data['BeginDate'] + timedelta(days=median_duration))
    merged_data['ContractDuration'] = merged_data['ContractDuration'].fillna(median_duration)

    # Convertir 'TotalCharges' a float y eliminar valores ausentes
    merged_data['TotalCharges'] = pd.to_numeric(merged_data['TotalCharges'], errors='coerce')
    merged_data.dropna(subset=['TotalCharges'], inplace=True)

    # Convertir columnas categóricas
    categorical_cols = ['Type', 'Gender', 'PaperlessBilling', 'PaymentMethod', 'Partner', 'Dependents', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                        'SeniorCitizen', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
    merged_data[categorical_cols] = merged_data[categorical_cols].astype('category')

    return merged_data

# ==============================
# 2. Exploratory Data Analysis (EDA)
# ==============================
def perform_eda(data):
    """Genera gráficos y métricas estadísticas."""
    print(data.describe())

    # Histograma de ContractDuration
    plt.figure(figsize=(10, 6))
    sns.histplot(data['ContractDuration'], kde=True, bins=30, color='green')
    plt.title('Distribución de ContractDuration')
    plt.xlabel('Duración del Contrato (días)')
    plt.ylabel('Frecuencia')
    plt.show()

    # Boxplot de MonthlyCharges
    plt.figure(figsize=(8, 5))
    sns.boxplot(data['MonthlyCharges'], color='blue')
    plt.title('Boxplot de Cargos Mensuales')
    plt.xlabel('MonthlyCharges USD')
    plt.show()

# ==============================
# 3. Preparación de datos para modelos
# ==============================
def prepare_data(df):
    """Prepara los datos para entrenamiento."""
    X = df.drop(columns=['Churn', 'ContractDuration', 'DurationSegment'])
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==============================
# 4. Entrenamiento y Optimización de Modelos
# ==============================
def optimize_xgb(X_train, y_train):
    """Optimiza XGBoost usando Grid Search."""
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='roc_auc', cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Mejores parámetros XGBoost:", grid_search.best_params_)
    return grid_search.best_estimator_

def optimize_catboost(X_train, y_train):
    """Optimiza CatBoost usando Grid Search."""
    param_grid = {
        'iterations': [200, 400],
        'learning_rate': [0.05, 0.1],
        'depth': [4, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bylevel': [0.6, 0.8, 1.0]
    }
    catboost_model = CatBoostClassifier(random_state=42, eval_metric='AUC', verbose=0)
    grid_search = GridSearchCV(catboost_model, param_grid, scoring='roc_auc', cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Mejores parámetros CatBoost:", grid_search.best_params_)
    return grid_search.best_estimator_

# ==============================
# 5. Evaluación de Modelos
# ==============================
def evaluate_model(model, X_test, y_test, model_name):
    """Evalúa el modelo con métricas clave y grafica la curva ROC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nEvaluación de {model_name}:")
    print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"Exactitud: {accuracy_score(y_test, y_pred):.3f}")

    # Gráfica de la Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC-ROC = {roc_auc_score(y_test, y_prob):.3f}")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title(f"Curva ROC para {model_name}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

# ==============================
# 6. Ejecución del Pipeline
# ==============================
def main():
    """Ejecuta el pipeline completo."""
    path = '/proyecto/datasets/final_provider/'
    contract, personal, internet, phone = load_data(path)
    
    merged_data = clean_data(contract, personal, internet, phone)
    merged_data.to_csv("dataset_limpio.csv", index=False)

    perform_eda(merged_data)

    X_train, X_test, y_train, y_test = prepare_data(merged_data)

    best_xgb = optimize_xgb(X_train, y_train)
    best_catboost = optimize_catboost(X_train, y_train)

    evaluate_model(best_xgb, X_test, y_test, "XGBoost")
    evaluate_model(best_catboost, X_test, y_test, "CatBoost")

if __name__ == "__main__":
    main()