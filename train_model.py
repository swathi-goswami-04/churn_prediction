import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
from scipy.stats import randint, uniform

# --- Configuration ---
RANDOM_STATE = 42
MLFLOW_EXPERIMENT_NAME = "Customer_Churn_Prediction"
MLFLOW_TRACKING_URI = "file:./mlruns" # Local MLflow tracking server
FINAL_MODEL_PATH = "artifacts/best_model.joblib"
TRAIN_SET_PATH = 'data/processed/train.csv'
VALIDATION_SET_PATH = 'data/processed/validation.csv'

def load_data(file_path):
    """Loads processed data and separates features (X) and target (y)."""
    df = pd.read_csv(file_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

def train_and_evaluate():
    """Main function to run the model training, tuning, and MLflow logging."""
    
    # 0. Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # 1. Load Data
    X_train, y_train = load_data(TRAIN_SET_PATH)
    X_val, y_val = load_data(VALIDATION_SET_PATH)

    print(f"Data Loaded: Train {X_train.shape}, Validation {X_val.shape}")

    # 2. Define Base Model and Hyperparameter Search Space
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=RANDOM_STATE
    )

    # Define a focused search space for Randomized Search
    # This is much faster than Grid Search and still effective
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }

    # 3. Randomized Hyperparameter Tuning (using Training Data)
    # Use ROC-AUC as the primary metric for optimization
    tuning = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=20, # Number of parameter settings that are sampled
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Start MLflow run to log all tuning results
    with mlflow.start_run(run_name="Tuning_Run_XGBoost") as run:
        print("Starting Randomized Search...")
        tuning.fit(X_train, y_train)

        best_params = tuning.best_params_
        best_score = tuning.best_score_
        best_estimator = tuning.best_estimator_
        
        # Log Tuning Results
        mlflow.log_params(best_params)
        mlflow.log_metric("tuning_best_cv_roc_auc", best_score)
        print(f"\nBest CV ROC-AUC Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")

        # 4. Final Evaluation (using Validation Data)
        
        # Predict probabilities on the Validation Set
        y_pred_proba = best_estimator.predict_proba(X_val)[:, 1]
        
        # Calculate Core Metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        
        # Calculate metrics that depend on a threshold (using default 0.5)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        # Log Final Validation Metrics
        mlflow.log_metric("val_roc_auc", roc_auc)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)

        # Print detailed report
        print("\n--- Final Validation Metrics ---")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("\nClassification Report:\n", classification_report(y_val, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
        
        # 5. Save Model Artifact and Log
        
        # Save the best model locally
        os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
        joblib.dump(best_estimator, FINAL_MODEL_PATH)
        
        # Log model with MLflow (essential for deployment)
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="model",
            registered_model_name="XGBoost_Churn_Predictor" # Optional: Register for versioning
        )
        
        print(f"\nBest model saved locally to: {FINAL_MODEL_PATH}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print("Model artifact logged to MLflow successfully.")

if __name__ == "__main__":
    train_and_evaluate()
    print("\n--- Phase 2 Complete ---")
    print("Next: Phase 3 (MLOps Pipeline) - Using the saved model artifact for deployment.")