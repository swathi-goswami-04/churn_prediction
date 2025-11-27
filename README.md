🚀 TELCO CUSTOMER CHURN PREDICTION MLOPS STACK

This project delivers an end-to-end Machine Learning Operations (MLOps) solution designed to predict customer churn risk for a telecommunications company. It consists of two primary services communicating via a modern API standard.

🏗️ Project Architecture
Goal: To serve a pre-trained XGBoost model and consume its predictions through a user-friendly interface.

Decoupled Services: The application is split into two distinct services that communicate via HTTP requests, ensuring scalability and maintainability .

1. Backend API (FastAPI)

Technology: FastAPI and Python.

Function: Hosts the trained ML model and data preprocessor (artifacts/).

Validation: Enforces strict data integrity using Pydantic schemas for input validation.

Local Endpoint: http://localhost:8000/predict (Prediction endpoint).

Documentation: Swagger UI available at http://localhost:8000/docs.

2. Frontend Web (Flask)

Technology: Flask and Jinja2.

Function: Provides the User Interface (UI) for customer data entry.

Role: Sends the collected data as a JSON payload to the FastAPI service and displays the calculated churn probability.

Local Endpoint: http://127.0.0.1:5000/ (Main application view).

🛠️ Local Setup and Run Instructions
Prerequisites
Python 3.9+

pip

Git

Artifacts: Ensure the artifacts/preprocessor.joblib and artifacts/best_model.joblib files are present.

Setup and Dependencies
Clone the repository:

Bash

git clone [<YOUR_REPO_URL>]
cd <YOUR_PROJECT_NAME>
Install dependencies for the Backend (use venv_backend):

Bash

python -m venv venv_backend
source venv_backend/bin/activate
pip install -r backend_requirements.txt
deactivate
Install dependencies for the Frontend (use venv_frontend):

Bash

python -m venv venv_frontend
source venv_frontend/bin/activate
pip install -r frontend_requirements.txt
deactivate
Running the Services
Start FastAPI Backend (Port 8000):

Bash

source venv_backend/bin/activate 
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Start Flask Frontend (Port 5000):

Bash

source venv_frontend/bin/activate 
python frontend_app.py
☁️ Deployment to Render
This process involves deploying the two services separately to Render as Web Services.

Update API Endpoint
Before deployment, update the API_URL variable in frontend_app.py to point to the live public URL of your deployed FastAPI service.

Python

# Change from local: API_URL = "http://localhost:8000/predict" 
# To Render: API_URL = "https://<YOUR-RENDER-BACKEND-NAME>.onrender.com/predict" 
Commit and push this change before deploying.

Deployment Configuration
FastAPI Backend Service:

Build Command: pip install -r backend_requirements.txt

Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT

Flask Frontend Service:

Build Command: pip install -r frontend_requirements.txt

Start Command: gunicorn frontend_app:app (Uses Gunicorn for stable production serving).

🔗 Repository Contents
app.py: FastAPI server logic.

frontend_app.py: Flask server logic.

index.html: Jinja2 web template.

artifacts/: Directory containing model files (.joblib).

backend_requirements.txt: Dependencies for the FastAPI service.

frontend_requirements.txt: Dependencies for the Flask service.
