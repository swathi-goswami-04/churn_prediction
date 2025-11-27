import os
import requests
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

# --- Configuration ---
# Point this to your running FastAPI model service endpoint
# The Dockerfile for the FastAPI app exposes port 8000
API_URL = "https://telco-churn-api-6h4y.onrender.com/predict" 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_secure_random_key_for_flash_messages' # Required for flash messages

# Define the fields required by the FastAPI/Pydantic model
FORM_FIELDS = [
    # General Info
    {'name': 'gender', 'type': 'select', 'options': ['Female', 'Male'], 'label': 'Gender'},
    {'name': 'SeniorCitizen', 'type': 'select', 'options': [0, 1], 'label': 'Senior Citizen (0=No, 1=Yes)'},
    {'name': 'Partner', 'type': 'select', 'options': ['Yes', 'No'], 'label': 'Has Partner'},
    {'name': 'Dependents', 'type': 'select', 'options': ['Yes', 'No'], 'label': 'Has Dependents'},
    
    # Service Info
    {'name': 'tenure', 'type': 'number', 'label': 'Tenure (Months)', 'placeholder': 'e.g., 24'},
    {'name': 'PhoneService', 'type': 'select', 'options': ['Yes', 'No'], 'label': 'Phone Service'},
    {'name': 'MultipleLines', 'type': 'select', 'options': ['Yes', 'No', 'No phone service'], 'label': 'Multiple Lines'},
    {'name': 'InternetService', 'type': 'select', 'options': ['DSL', 'Fiber optic', 'No'], 'label': 'Internet Service'},
    {'name': 'PaperlessBilling', 'type': 'select', 'options': ['Yes', 'No'], 'label': 'Paperless Billing'},
    {'name': 'Contract', 'type': 'select', 'options': ['Month-to-month', 'One year', 'Two year'], 'label': 'Contract Type'},
    {'name': 'PaymentMethod', 'type': 'select', 'options': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 'label': 'Payment Method'},
    
    # Internet Services
    {'name': 'OnlineSecurity', 'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'label': 'Online Security'},
    {'name': 'OnlineBackup', 'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'label': 'Online Backup'},
    {'name': 'DeviceProtection', 'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'label': 'Device Protection'},
    {'name': 'TechSupport', 'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'label': 'Tech Support'},
    {'name': 'StreamingTV', 'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'label': 'Streaming TV'},
    {'name': 'StreamingMovies', 'type': 'select', 'options': ['Yes', 'No', 'No internet service'], 'label': 'Streaming Movies'},

    # Charges
    {'name': 'MonthlyCharges', 'type': 'number', 'label': 'Monthly Charges ($)', 'placeholder': 'e.g., 70.35'},
    {'name': 'TotalCharges', 'type': 'number', 'label': 'Total Charges ($)', 'placeholder': 'e.g., 1889.5'}
]


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    
    if request.method == 'POST':
        try:
            # 1. Collect and format data from the form
            form_data = request.form.to_dict()
            # --- FINAL CORRECT CODE: REPLACE ENTIRE LOOP WITH THIS ---

            data_payload = {}
            for key, value in form_data.items():
                
                # Strip any leading/trailing whitespace
                cleaned_value = value.strip()
                cleaned_value_lower = cleaned_value.lower()

                if key == 'tenure':
                    # tenure is an integer. Handle empty string as 0.
                    data_payload[key] = int(cleaned_value or 0)
                
                elif key in ['MonthlyCharges', 'TotalCharges']:
                    # Charges are floats/numbers. Handle empty string as 0.0.
                    data_payload[key] = float(cleaned_value or 0.0)
                
                elif key == 'SeniorCitizen':
                    # FIX: Explicitly handle the string 'Yes' or 'No' and the integer strings '1' or '0'
                    if cleaned_value_lower in ['yes', '1']:
                        data_payload[key] = 1
                    else:
                        # This covers 'No', '0', or any other unexpected string, defaulting to 0
                        data_payload[key] = 0
                
                else:
                    # All other fields are categorical strings.
                    data_payload[key] = cleaned_value
        
# --- END REPLACEMENT ---

# --- END REPLACEMENT ---

            # 2. Send request to the FastAPI prediction endpoint
            response = requests.post(API_URL, json=data_payload)
            response.raise_for_status() # Raise exception for HTTP errors (4xx or 5xx)
            
            # 3. Process the API response
            prediction_result = response.json()
            
            if 'error' in prediction_result:
                flash(f"API Error: {prediction_result['error']}", 'danger')
            else:
                # Store the successful result to be displayed in the template
                # Use a specific color based on the prediction
                if prediction_result['prediction'] == 1:
                    prediction_result['color'] = 'danger' # Red for Churn
                else:
                    prediction_result['color'] = 'success' # Green for No Churn

        except requests.exceptions.ConnectionError:
            flash(f"Connection Error: Could not connect to the FastAPI service at {API_URL}. Ensure the service is running!", 'danger')
        except requests.exceptions.RequestException as e:
            flash(f"An unexpected API request error occurred: {e}", 'danger')
        except ValueError as e:
            flash(f"Data conversion error (check numeric fields): {e}", 'danger')
        except Exception as e:
            flash(f"An unknown error occurred: {e}", 'danger')

    # Render the template, passing the form fields and the result
    return render_template('index.html', form_fields=FORM_FIELDS, prediction_result=prediction_result)

if __name__ == '__main__':
    # Flask will run on port 5000 by default
    app.run(debug=True, port=5000)