import argparse
import pandas as pd
import time
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def load_data(url):
    """
    Load dataset from the given URL.
    """
    try:
        df = pd.read_csv(url)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Split the dataframe into X (features) and y (target).
    """
    print("⚙️ Preprocessing data...")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_pipeline():
    """
    Create a machine learning pipeline with StandardScaler and RandomForestRegressor.
    """
    return Pipeline(steps=[
        ("standard_scaler", StandardScaler()),
        ("Random_Forest", RandomForestRegressor())
    ])

def train_model(pipe, X_train, y_train, param_grid, cv=2):
    """
    Train the model using GridSearchCV.
    """
    print(f"🏋️ Training model with grid: {param_grid}")
    model = GridSearchCV(pipe, param_grid, verbose=0, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    return model

def run_training(args, param_grid, DATA_URL):
    """
    Fonction principale d'entraînement (sans gestion de run MLflow)
    """
    start_time = time.time()

    # Load & Preprocess
    df = load_data(DATA_URL)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train
    pipe = create_pipeline()
    model = train_model(pipe, X_train, y_train, param_grid)

    # Logging
    best_score = model.best_score_
    test_score = model.score(X_test, y_test)
    
    print(f"📊 Train CV Score: {best_score:.4f}")
    print(f"📊 Test Score:     {test_score:.4f}")

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("criterion", args.criterion)
    
    mlflow.log_metric("train_cv_score", best_score)
    mlflow.log_metric("test_score", test_score)
    mlflow.log_metric("training_time", time.time() - start_time)

    print("💾 Saving model to MLflow...")
    mlflow.sklearn.log_model(
        sk_model=model.best_estimator_,
        artifact_path="model",
        registered_model_name="random_forest_regressor"
    )
    
    print("✅ Training Complete.")

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Parse ONLY Model Hyperparameters
    parser = argparse.ArgumentParser(description="Random Forest Training Script")
    parser.add_argument("--n_estimators", type=int, default=20)
    parser.add_argument("--criterion", type=str, default="squared_error")
    parser.add_argument("--experiment_name", type=str, default="california_housing")
    args = parser.parse_args()
    
    # Configuration MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    # Ne pas appeler set_experiment si MLflow Projects a déjà défini l'expérience
    if not os.environ.get("MLFLOW_EXPERIMENT_ID"):
        mlflow.set_experiment(args.experiment_name)
        print(f"🚀 Starting MLflow Run in experiment: {args.experiment_name}")
    else:
        print(f"🚀 Using MLflow experiment ID: {os.environ.get('MLFLOW_EXPERIMENT_ID')}")

    # 3. Configuration
    DATA_URL = "https://julie-2-next-resources.s3.eu-west-3.amazonaws.com/full-stack-full-time/linear-regression-ft/californian-housing-market-ft/california_housing_market.csv"
    
    param_grid = {
        "Random_Forest__n_estimators": [args.n_estimators],
        "Random_Forest__criterion": [args.criterion]
    }

    # 4. Start Run - vérifie si MLFLOW_RUN_ID est défini par MLflow Projects
    if os.environ.get("MLFLOW_RUN_ID"):
        # MLflow Projects a défini un run via variable d'environnement
        print(f"📌 Using existing MLflow run: {os.environ.get('MLFLOW_RUN_ID')}")
        run_training(args, param_grid, DATA_URL)
    else:
        # Pas de run défini, on en crée un
        print("📌 Creating new MLflow run...")
        with mlflow.start_run():
            run_training(args, param_grid, DATA_URL)
