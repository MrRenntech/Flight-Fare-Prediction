# ✈️ Flight Fare Prediction

This project predicts the price of airline tickets based on features like source, destination, duration, and number of stops. It uses a **Random Forest Regressor** model served via a **Flask** web application.

## 🚀 Quick Start (Windows)

1.  **Install Prerequisites**:
    Double-click `install_prereq.bat` to install Python dependencies.
    *(Requires Python to be installed and added to PATH)*.

2.  **Train Model (Optional)**:
    If you want to retrain the model from scratch, run:
    ```bash
    python "Flight model.py"
    ```
    This generates `flight_rf.pkl` and `columns.json`.

3.  **Launch App**:
    Double-click `launch.bat`.
    Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 📁 Project Structure

```bash
Flight-Fare-Prediction/
├── main.py                 # Flask backend (Prediction logic)
├── Flight model.py         # Model training script
├── flight_rf.pkl           # Trained ML model
├── columns.json            # Model feature names (MetaData)
├── requirements.txt        # Dependencies
├── HowItWorks.txt          # Detailed explanation of logic
├── launch.bat              # Shortcut to run app
├── install_prereq.bat      # Method to install libs
├── Data_Train.xlsx         # Dataset
└── templates/
    └── home.html           # UI Template
```

## 🛠️ Tech Stack
-   **Python 3.x**
-   **Flask**
-   **Scikit-Learn**
-   **Pandas**

## 📝 Changelog
See `CHANGELOG.md` for recent updates.
