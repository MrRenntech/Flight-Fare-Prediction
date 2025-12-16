"""
================================================================================
FLIGHT FARE PREDICTION - MODEL TRAINING SCRIPT
================================================================================
This script trains a Random Forest Regressor model to predict flight fares.
It handles data loading, preprocessing, feature engineering, model training,
and artifact saving.

Libraries Used:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- pickle: Model serialization
- json: Metadata storage
- scikit-learn: Machine learning algorithms

Author: Antigravity AI
Last Updated: December 2024
================================================================================
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_data(file_path):
    """
    Loads the flight fare dataset from an Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file containing training data
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset with null values removed
    """
    print(f"Loading data from {file_path}...")
    data = pd.read_excel(file_path)
    # Remove any rows with missing values
    data.dropna(inplace=True)
    return data

def preprocess_data(data):
    """
    Preprocesses the flight fare data by:
    1. Extracting date/time features
    2. Parsing duration strings
    3. Encoding categorical variables
    4. Dropping unnecessary columns
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw flight data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data ready for model training
    """
    print("Preprocessing data...")
    
    # ========== DATE OF JOURNEY PROCESSING ==========
    # Extract day and month from the journey date
    if "Date_of_Journey" in data.columns:
        data["Journey_day"] = pd.to_datetime(data.Date_of_Journey, format="%d/%m/%Y").dt.day
        data["Journey_month"] = pd.to_datetime(data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
        data.drop(["Date_of_Journey"], axis=1, inplace=True)

    # ========== DEPARTURE TIME PROCESSING ==========
    # Extract hour and minute from departure time
    if "Dep_Time" in data.columns:
        data["Dep_hour"] = pd.to_datetime(data["Dep_Time"]).dt.hour
        data["Dep_min"] = pd.to_datetime(data["Dep_Time"]).dt.minute
        data.drop(["Dep_Time"], axis=1, inplace=True)

    # ========== ARRIVAL TIME PROCESSING ==========
    # Extract hour and minute from arrival time
    if "Arrival_Time" in data.columns:
        data["Arrival_hour"] = pd.to_datetime(data.Arrival_Time).dt.hour
        data["Arrival_min"] = pd.to_datetime(data.Arrival_Time).dt.minute
        data.drop(["Arrival_Time"], axis=1, inplace=True)

    # ========== DURATION PROCESSING ==========
    # Parse duration strings like "2h 50m" into separate hour and minute columns
    if "Duration" in data.columns:
        duration = list(data["Duration"])
        
        # Standardize duration format
        for i in range(len(duration)):
            # If duration only has hours or only minutes, add the missing component
            if len(duration[i].split()) != 2:
                if "h" in duration[i]:
                    duration[i] = duration[i].strip() + " 0m"   # Add 0 minutes
                else:
                    duration[i] = "0h " + duration[i]           # Add 0 hours
        
        # Extract hours and minutes separately
        duration_hours = []
        duration_mins = []
        for i in range(len(duration)):
            duration_hours.append(int(duration[i].split(sep = "h")[0]))
            duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))
            
        # Add new columns to dataframe
        data["Duration_hours"] = duration_hours
        data["Duration_mins"] = duration_mins
        data.drop(["Duration"], axis=1, inplace=True)

    # ========== CATEGORICAL ENCODING ==========
    
    # One-Hot Encode Airline (Nominal categorical variable)
    Airline = data[["Airline"]]
    Airline = pd.get_dummies(Airline, drop_first=True)

    # One-Hot Encode Source City (Nominal categorical variable)
    Source = data[["Source"]]
    Source = pd.get_dummies(Source, drop_first=True)

    # One-Hot Encode Destination City (Nominal categorical variable)
    Destination = data[["Destination"]]
    Destination = pd.get_dummies(Destination, drop_first=True)

    # ========== DROP UNNECESSARY COLUMNS ==========
    # Route and Additional_Info don't add value to the model
    data.drop(["Route", "Additional_Info"], axis=1, inplace=True)

    # ========== LABEL ENCODE TOTAL STOPS ==========
    # This is ordinal data (order matters: 0 stops < 1 stop < 2 stops)
    data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

    # ========== CONCATENATE ALL FEATURES ==========
    # Combine original data with encoded categorical variables
    data_train = pd.concat([data, Airline, Source, Destination], axis=1)
    data_train.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)
    
    return data_train

def train_model(X, y):
    """
    Trains a Random Forest Regressor model on the provided features and target.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix (independent variables)
    y : pandas.Series
        Target variable (flight price)
        
    Returns:
    --------
    RandomForestRegressor
        Trained model
    """
    print("Training model...")
    
    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest model
    reg_rf = RandomForestRegressor()
    reg_rf.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = reg_rf.predict(X_test)
    
    # ========== MODEL EVALUATION ==========
    # Print performance metrics
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 Score:', metrics.r2_score(y_test, y_pred))
    
    return reg_rf

def save_artifacts(model, columns, model_path="flight_rf.pkl", columns_path="columns.json"):
    """
    Saves the trained model and feature column names to disk.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model to save
    columns : list
        List of feature column names
    model_path : str
        Path where model will be saved (default: "flight_rf.pkl")
    columns_path : str
        Path where column names will be saved (default: "columns.json")
    """
    print(f"Saving artifacts to {model_path} and {columns_path}...")
    
    # Save model using pickle (binary serialization)
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
        
    # Save column names using JSON (human-readable format)
    with open(columns_path, 'w') as f:
        json.dump(columns, f)

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Step 1: Load data from Excel file
    train_data = load_data("Data_Train.xlsx")
    
    # Step 2: Preprocess the data (feature engineering, encoding)
    processed_data = preprocess_data(train_data)
    
    # Step 3: Prepare column list (removing target variable 'Price')
    columns = processed_data.columns.tolist()
    if "Price" in columns:
        columns.remove("Price")
        
    # Step 4: Split into features (X) and target (y)
    X = processed_data.drop("Price", axis=1)
    y = processed_data["Price"]
    
    # Step 5: Train the model
    model = train_model(X, y)
    
    # Step 6: Save model and metadata
    save_artifacts(model, columns)
    print("Done!")