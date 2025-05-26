# app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    model_path = 'random_forest_model.pkl'
    model = joblib.load(model_path)

    # Step 1: Get JSON input data
    data = request.get_json()

    features = [
            float(data["X1"]), float(data["X2"]), float(data["X3"]),
            float(data["X4"]), float(data["X5"]), float(data["X6"]),
            float(data["X7"]), float(data["X8"]), float(data["X9"]),
            float(data["X10"]), float(data["X11"]), float(data["X12"]),
            float(data["X13"]), float(data["X14"]), float(data["X15"]),
            float(data["X16"]), float(data["X17"]), float(data["X18"]),
            int(data["year"])
        ]

    # Step 3: Create DataFrame
    column_names = [
        'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9',
        'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'year'
    ]

    df = pd.DataFrame([features], columns=column_names)

    # Step 4: Select only the important features for prediction
    important_features = ['X1', 'X3', 'X4', 'X6', 'X7', 'X8', 'X11', 'X12', 'X15', 'X17', 'year']
    df = df[important_features]

    # Step 5: Run prediction
    prediction = model.predict(df)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True, port = 5000)