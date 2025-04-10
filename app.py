from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')


# Load the saved scikit-learn pipeline
pipeline = joblib.load('detector.joblib')

@app.route('/')
def home():
    template_dir = os.path.abspath('./')
    return render_template('index.html', template_dir=template_dir)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Get the user input from the form
    input_features = [float(x) for x in request.form.values()]

    # Convert the input features to a Pandas DataFrame
    input_df = pd.DataFrame([input_features], columns=['concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst', 'perimeter_mean', 'area_worst', 'radius_mean'])

    # Use the pipeline to make a prediction
    prediction = pipeline.predict(input_df)

    # Convert the prediction to a string
    if prediction == 1:
        result = 'Malignant'
        prediction_color = 'malignant'
    else:
        result = 'Benign'
        prediction_color = 'benign'

    # Render the index page with the prediction
    template_dir = os.path.abspath('./')
    return render_template('index.html', prediction=result, prediction_color=prediction_color, template_dir=template_dir)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)