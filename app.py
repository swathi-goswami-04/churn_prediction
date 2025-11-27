import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# --- Configuration & Initialization ---
PREPROCESSOR_MODEL_PATH = "artifacts/preprocessor.joblib"
FINAL_MODEL_PATH = "artifacts/best_model.joblib"

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn using a pre-trained XGBoost model.",
    version="1.0.0"
)

# Load the trained artifacts
try:
    preprocessor = joblib.load(PREPROCESSOR_MODEL_PATH)
    model = joblib.load(FINAL_MODEL_PATH)
    print("Model and Preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    preprocessor = None
    model = None

# --- Pydantic Data Model (Input Schema) ---
class Customer(BaseModel):
    gender: Literal['Female', 'Male'] = Field(..., description="Customer's gender.")
    SeniorCitizen: int = Field(..., description="1 if customer is a senior citizen, 0 otherwise.")
    Partner: Literal['Yes', 'No'] = Field(..., description="Whether the customer has a partner.")
    Dependents: Literal['Yes', 'No'] = Field(..., description="Whether the customer has dependents.")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company.")
    PhoneService: Literal['Yes', 'No'] = Field(..., description="Whether the customer has phone service.")
    MultipleLines: Literal['Yes', 'No', 'No phone service'] = Field(..., description="Whether the customer has multiple lines.")
    InternetService: Literal['DSL', 'Fiber optic', 'No'] = Field(..., description="Customer's internet service provider.")
    OnlineSecurity: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Whether the customer has online security.")
    OnlineBackup: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Whether the customer has online backup.")
    DeviceProtection: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Whether the customer has device protection.")
    TechSupport: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Whether the customer has technical support.")
    StreamingTV: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Whether the customer has streaming TV.")
    StreamingMovies: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Whether the customer has streaming movies.")
    Contract: Literal['Month-to-month', 'One year', 'Two year'] = Field(..., description="The customer’s current contract type.")
    PaperlessBilling: Literal['Yes', 'No'] = Field(..., description="Whether the customer has paperless billing.")
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'] = Field(..., description="The customer’s payment method.")
    MonthlyCharges: float = Field(..., description="The amount charged to the customer monthly.")
    TotalCharges: float = Field(..., description="The total amount charged to the customer.")

# --- Feature Engineering Function (Re-implementation of Phase 1 logic) ---
def apply_feature_engineering(df):
    """
    Applies the same custom feature engineering steps used in Phase 1 (train_model.py).
    """
    # 1. Handle 'No internet service' and 'No phone service'
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']:
        df[col] = df[col].replace('No internet service', 'No')
        df[col] = df[col].replace('No phone service', 'No')

    # 2. Feature Engineering: Average Monthly Charge
    df['Avg_Monthly_Charge'] = np.where(df['tenure'] > 0, 
                                        df['TotalCharges'] / df['tenure'], 
                                        0)

    # 3. Feature Engineering: Tenure Group (Categorical)
    bins=[-1, 12, 24, 60, 100]
    labels=['0-1 Yr', '1-2 Yrs', '2-5 Yrs', '5+ Yrs']
    # .astype(str) handles potential NaN/None from pandas.cut
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=True).astype(str)
    
    # 4. Drop TotalCharges (must be done AFTER Avg_Monthly_Charge calculation)
    df.drop('TotalCharges', axis=1, inplace=True)
    
    return df

# --- API Endpoint ---
@app.post("/predict")
async def predict_churn(customer_data: Customer):
    """
    Predicts the probability of customer churn (1) vs non-churn (0).
    """
    if preprocessor is None or model is None:
        return {"error": "Model artifacts not loaded. Check server logs."}

    # 1. Convert Pydantic model to DataFrame
    raw_data = customer_data.model_dump()
    raw_df = pd.DataFrame([raw_data])
    
    # 2. Apply feature engineering
    engineered_df = apply_feature_engineering(raw_df.copy())
    
    # 3. Apply preprocessing pipeline (Scaling and Encoding)
    try:
        processed_data = preprocessor.transform(engineered_df)
    except Exception as e:
        return {"error": f"Preprocessing failed: {e}", "details": "Check input data against required categories."}

    # 4. Generate prediction
    # Get the churn probability (class 1)
    churn_proba = model.predict_proba(processed_data)[:, 1][0]
    
    # Apply a prediction threshold (e.g., 0.5 for binary decision)
    binary_prediction = 1 if churn_proba >= 0.5 else 0

    # 5. FIX: Explicitly cast NumPy types to native Python types for JSON serialization
    return {
        "churn_probability": float(churn_proba), # Convert numpy.float32 to float
        "prediction": int(binary_prediction),       # Convert numpy.int32/int64 to int
        "interpretation": "Customer is predicted to CHURN" if binary_prediction == 1 else "Customer is predicted NOT to churn"
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "API is running and models are loaded."}