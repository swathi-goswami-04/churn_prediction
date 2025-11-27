# 🚀 END-TO-END TELCO CHURN PREDICTION MLOPS STACK 📊

This project implements a complete Machine Learning Operations (MLOps) pipeline for predicting customer churn risk. It features a robust, decoupled architecture where a production-grade machine learning model is served via a high-performance API.

-----

## ✨ Key Technologies

  * **Backend & API:** FastAPI, Pydantic
  * **Frontend & UI:** Flask, Jinja2, Bootstrap
  * **Machine Learning:** XGBoost, Scikit-learn, Joblib, Pandas
  * **Deployment:** Render (Cloud Platform)

-----

## 🏗️ Architecture Overview

The system is designed with a service-oriented architecture to ensure scalability and independent deployment .

### 1\. Backend Prediction API (FastAPI)

  * **Role:** Model serving and inference.
  * **Features:**
      * Loads the trained model (`best_model.joblib`) and preprocessor (`preprocessor.joblib`) from the `artifacts/` directory.
      * Enforces strict data validation using **Pydantic** schemas.
      * **Local Endpoint:** `http://localhost:8000/predict`

### 2\. Frontend Web Application (Flask)

  * **Role:** User interface and API client.
  * **Features:**
      * Provides a clean, aesthetic web form for user data entry.
      * Collects and formats the data into a JSON payload.
      * Sends the JSON payload to the FastAPI backend and displays the churn probability result.
      * **Local Endpoint:** `http://127.0.0.1:5000/`

-----

## 🛠️ Local Setup and Execution

### Prerequisites

  * Python 3.9+
  * Git
  * All model artifacts present in the `artifacts/` folder.

### Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone <YOUR_REPO_URL>
    cd <YOUR_PROJECT_NAME>
    ```

2.  **Install Backend Dependencies** (Use `venv_backend`):

    ```bash
    python -m venv venv_backend
    source venv_backend/bin/activate
    pip install -r backend_requirements.txt
    deactivate
    ```

3.  **Install Frontend Dependencies** (Use `venv_frontend`):

    ```bash
    python -m venv venv_frontend
    source venv_frontend/bin/activate
    pip install -r frontend_requirements.txt
    deactivate
    ```

### Running the Services

  * **Start FastAPI Backend:**
    ```bash
    source venv_backend/bin/activate 
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```
  * **Start Flask Frontend:**
    ```bash
    source venv_frontend/bin/activate 
    python frontend_app.py
    ```

-----

## ☁️ Cloud Deployment (Render)

The application is configured for deployment as two separate **Web Services** on Render for robust production handling.

### Deployment Preparation

  * **Update Endpoint:** Prior to pushing for deployment, ensure the `API_URL` variable in **`frontend_app.py`** is updated to the live public URL of your FastAPI service (e.g., `https://my-telco-api.onrender.com/predict`).
  * **Commit:** Commit this change and push to your main branch.

### Render Configuration Summary

  * **FastAPI Backend Service:**
      * **Build Command:** `pip install -r backend_requirements.txt`
      * **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
  * **Flask Frontend Service:**
      * **Build Command:** `pip install -r frontend_requirements.txt`
      * **Start Command:** `gunicorn frontend_app:app` (Recommended Gunicorn for production Flask serving).

-----

## 📂 Repository Structure

  * `app.py`: FastAPI application, model loading, and prediction logic.
  * `frontend_app.py`: Flask application, UI logic, and API client.
  * `index.html`: Web interface template (Jinja2).
  * `artifacts/`: Contains saved model artifacts (`.joblib` files).
  * `backend_requirements.txt`: Dependencies for the FastAPI service.
  * `frontend_requirements.txt`: Dependencies for the Flask service.
