"""
================================================================================
FLIGHT FARE PREDICTION - FLASK WEB APPLICATION
================================================================================
This Flask application serves the trained Random Forest model as a web API.
Users can input flight details through a web form and receive fare predictions.

Key Features:
- Dynamic feature alignment using columns.json
- One-Hot Encoding handled automatically
- Error handling for invalid inputs

Libraries Used:
- Flask: Web framework
- flask-cors: Cross-Origin Resource Sharing support
- pickle: Model loading
- pandas: Data manipulation
- json: Metadata loading

Author: Antigravity AI
Last Updated: December 2024
================================================================================
"""

from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd
import json

# Initialize Flask application
app = Flask(__name__)

# ========== LOAD MODEL AND METADATA ==========
print("Loading model and columns...")
# Load trained Random Forest model from pickle file
model = pickle.load(open("flight_rf.pkl", "rb"))
# Load feature column names from JSON file
# This ensures input features match what the model expects
with open("columns.json", "r") as f:
    model_columns = json.load(f)
print("Model and columns loaded.")

@app.route("/")
@cross_origin()
def home():
    """
    Renders the home page with the prediction form.
    
    Returns:
    --------
    HTML template
        The main page with input fields
    """
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    """
    Handles prediction requests from the web form.
    Processes user input, aligns features with model expectations,
    and returns the predicted flight fare.
    
    Returns:
    --------
    HTML template
        Home page with prediction result displayed
    """
    if request.method == "POST":
        try:
            # ========== EXTRACT FORM DATA ==========
            dep_time = request.form["Dep_Time"]
            arrival_time = request.form["Arrival_Time"]
            
            # ========== PROCESS DATE AND TIME FEATURES ==========
            # Extract journey day and month from departure datetime
            journey_day = int(pd.to_datetime(dep_time, format="%Y-%m-%dT%H:%M").day)
            journey_month = int(pd.to_datetime(dep_time, format="%Y-%m-%dT%H:%M").month)
            
            # Extract departure hour and minute
            dep_hour = int(pd.to_datetime(dep_time, format="%Y-%m-%dT%H:%M").hour)
            dep_min = int(pd.to_datetime(dep_time, format="%Y-%m-%dT%H:%M").minute)
            
            # Extract arrival hour and minute
            arrival_hour = int(pd.to_datetime(arrival_time, format="%Y-%m-%dT%H:%M").hour)
            arrival_min = int(pd.to_datetime(arrival_time, format="%Y-%m-%dT%H:%M").minute)
            
            # Calculate flight duration
            # Note: This is a simplified calculation (doesn't account for overnight flights)
            dur_hour = abs(arrival_hour - dep_hour)
            dur_min = abs(arrival_min - dep_min)
            
            # Extract total stops (already in numeric format from form)
            total_stops = int(request.form["stops"])
            
            # ========== INITIALIZE FEATURE DATAFRAME ==========
            # Create a DataFrame with all model features set to 0
            # This ensures we have all columns the model expects
            input_data = pd.DataFrame(0, index=[0], columns=model_columns)
            
            # ========== SET NUMERICAL FEATURES ==========
            input_data["Total_Stops"] = total_stops
            input_data["Journey_day"] = journey_day
            input_data["Journey_month"] = journey_month
            input_data["Dep_hour"] = dep_hour
            input_data["Dep_min"] = dep_min
            input_data["Arrival_hour"] = arrival_hour
            input_data["Arrival_min"] = arrival_min
            input_data["Duration_hours"] = dur_hour
            input_data["Duration_mins"] = dur_min

            # ========== HANDLE CATEGORICAL FEATURES ==========
            # The model expects One-Hot Encoded columns like:
            # "Airline_Jet Airways", "Source_Delhi", "Destination_Cochin"
            
            airline = request.form['airline']
            source = request.form['Source']
            destination = request.form['Destination']
            
            # Construct expected column names
            airline_col = f"Airline_{airline}"
            source_col = f"Source_{source}"
            destination_col = f"Destination_{destination}"
            
            # Set corresponding columns to 1 if they exist in model features
            # If not found, it means that category was the "dropped" reference category
            if airline_col in model_columns:
                input_data[airline_col] = 1
                
            if source_col in model_columns:
                input_data[source_col] = 1
                
            if destination_col in model_columns:
                input_data[destination_col] = 1

            # ========== MAKE PREDICTION ==========
            # Pass aligned feature DataFrame to model
            prediction = model.predict(input_data)
            # Round to 2 decimal places for display
            output = round(prediction[0], 2)
            
            # Return result to template
            return render_template('home.html', prediction_text="Your Flight price is Rs. {}".format(output))
            
        except Exception as e:
            # Handle any errors gracefully
            return render_template('home.html', prediction_text="Error in prediction: {}".format(str(e)))

    # For GET requests, just render the form
    return render_template("home.html")


# ========== RUN APPLICATION ==========
if __name__ == "__main__":
    # Run Flask development server
    # debug=True enables auto-reload and detailed error messages
    app.run(debug=True)