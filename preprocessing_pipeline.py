import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os

# --- Configuration ---
RAW_DATA_PATH = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
TRAIN_SET_PATH = 'data/processed/train.csv'
VALIDATION_SET_PATH = 'data/processed/validation.csv'
TEST_SET_PATH = 'data/processed/test.csv'
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
PREPROCESSOR_MODEL_PATH = 'artifacts/preprocessor.joblib'

def load_data(file_path):
    """Loads the dataset and drops the non-predictive customerID column."""
    df = pd.read_csv(file_path)
    df.drop('customerID', axis=1, inplace=True)
    print(f"Original Data Loaded. Shape: {df.shape}")
    return df

def feature_engineering(df):
    """
    Performs data type cleaning and custom feature engineering steps.
    """
    # 1. Handle TotalCharges missing/spaces issue
    # Coerce to numeric, turning spaces/invalid values into NaN, then fill with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # 2. Encode Target Variable (Churn)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 3. Handle 'No internet service' and 'No phone service' for consistent encoding
    # Replace these values in their respective columns to simplify encoding later
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']:
        df[col] = df[col].replace('No internet service', 'No')
        df[col] = df[col].replace('No phone service', 'No')

    # 4. Feature Engineering: Average Monthly Charge
    # Create Avg_Monthly_Charge, handling tenure=0 (which now has TotalCharges=0)
    df['Avg_Monthly_Charge'] = np.where(df['tenure'] > 0, 
                                        df['TotalCharges'] / df['tenure'], 
                                        0)

    # 5. Feature Engineering: Tenure Group (Categorical)
    df['Tenure_Group'] = pd.cut(df['tenure'],
                                bins=[-1, 12, 24, 60, 100],
                                labels=['0-1 Yr', '1-2 Yrs', '2-5 Yrs', '5+ Yrs'])
    
    # Drop TotalCharges as it's highly collinear
    df.drop('TotalCharges', axis=1, inplace=True)
    
    return df

def create_and_fit_pipeline(X_train):
    """
    Defines and fits the ColumnTransformer and Pipeline for data preprocessing
    on the training data.
    """
    # --- Define Feature Groups ---
    
    # Numerical Features (need scaling)
    numerical_features = ['tenure', 'MonthlyCharges', 'Avg_Monthly_Charge']
    
    # Binary Features (Yes/No or Male/Female, need OrdinalEncoder for 1/0 conversion)
    binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'PaperlessBilling', 'SeniorCitizen'] 
    
    # Categorical Features (Nominal, need One-Hot Encoding)
    nominal_features = ['InternetService', 'Contract', 'PaymentMethod', 'StreamingTV', 
                        'StreamingMovies', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'Tenure_Group']

    # --- Preprocessing Steps (Transformers) ---
    
    # 1. Numerical Pipeline: Scale numerical features
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # 2. Ordinal Pipeline (for binary features): Convert categorical strings to 0/1 and scale
    binary_pipeline = Pipeline([
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('std', StandardScaler())
    ])
    
    # 3. Categorical Pipeline: One-Hot Encode nominal features
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine all transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, nominal_features),
            ('bin', binary_pipeline, binary_features)
        ],
        remainder='drop' # Drop any columns not specified
    )
    
    # Fit the preprocessor ONLY on the training data
    preprocessor.fit(X_train)
    
    return preprocessor

def split_and_save_data(df):
    """
    Splits the data into Train, Validation, and Test sets using stratified sampling,
    applies the preprocessing pipeline, and saves the results along with the fitted pipeline.
    """
    
    # Separate features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 1. Split into Training (60%) and Holdout (40%)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, 
        test_size=VALIDATION_SIZE + TEST_SIZE, # 0.4
        random_state=RANDOM_STATE, 
        stratify=y 
    )

    # 2. Split Holdout (40%) into Validation (20%) and Test (20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, 
        test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE), # 0.2 / 0.4 = 0.5
        random_state=RANDOM_STATE, 
        stratify=y_holdout
    )
    
    # 3. Create and fit the Scikit-learn preprocessing pipeline on X_train
    preprocessor = create_and_fit_pipeline(X_train)
    
    # Transform all three sets
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert back to DataFrame for easy saving/loading
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Combine X and y for saving (a common practice)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_df = X_train_df.assign(Churn=y_train.values)
    val_df = X_val_df.assign(Churn=y_val.values)
    test_df = X_test_df.assign(Churn=y_test.values)

    # 4. Save Processed Data and Pipeline
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    train_df.to_csv(TRAIN_SET_PATH, index=False)
    val_df.to_csv(VALIDATION_SET_PATH, index=False)
    test_df.to_csv(TEST_SET_PATH, index=False)
    
    joblib.dump(preprocessor, PREPROCESSOR_MODEL_PATH)

    print("--- Data Split Summary ---")
    print(f"Training Set Shape: {train_df.shape}")
    print(f"Validation Set Shape: {val_df.shape}")
    print(f"Test Set Shape: {test_df.shape}")
    print("\nProcessed data and fitted preprocessor pipeline saved successfully.")
    
    return preprocessor

if __name__ == "__main__":
    # Ensure the input file is in the same directory as this script, or change RAW_DATA_PATH
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: The file '{RAW_DATA_PATH}' was not found. Please ensure it is in the current directory.")
    else:
        data = load_data(RAW_DATA_PATH)
        data_processed = feature_engineering(data.copy())
        split_and_save_data(data_processed)
        print("\n--- Phase 1 Complete ---")
        print("Data is ready for Phase 2: Model Training.")