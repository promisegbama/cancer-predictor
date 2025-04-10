from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__, template_folder='templates')


# Load the saved scikit-learn pipeline
pipeline = joblib.load('detector.joblib')

@app.route('/')
def home():
    template_dir = os.path.abspath('./')
    return render_template('index.html', template_dir=template_dir)

@app.route('/log-unusual-inputs', methods=['POST'])
def log_unusual_inputs():
    data = request.json
    log_path = 'unusual_inputs_log.json'

    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            json.dump([], f)

    with open(log_path, 'r+') as f:
        logs = json.load(f)
        logs.append(data)
        f.seek(0)
        json.dump(logs, f, indent=2)

    return {'status': 'logged'}, 200


from datetime import datetime

user_feedback = None
user_feedback_time = None

@app.route('/comment', methods=['POST'])
def comment():
    global user_feedback, user_feedback_time
    feedback = request.form.get('feedback')
    if feedback and not user_feedback:
        user_feedback = feedback
        user_feedback_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    return render_template("index.html", user_feedback=user_feedback, user_feedback_time=user_feedback_time)



@app.route('/predict', methods=['POST'])
def predict():
    input_values = [
        float(request.form['concave points_worst']),
        float(request.form['perimeter_worst']),
        float(request.form['concave points_mean']),
        float(request.form['radius_worst']),
        float(request.form['perimeter_mean']),
        float(request.form['area_worst']),
        float(request.form['radius_mean'])
    ]
    input_df = pd.DataFrame([input_values], columns=[
        'concave points_worst', 'perimeter_worst', 'concave points_mean',
        'radius_worst', 'perimeter_mean', 'area_worst', 'radius_mean'
    ])

    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][int(prediction)] * 100
    label = 'Malignant' if prediction == 1 else 'Benign'
    color = 'malignant' if prediction == 1 else 'benign'

    return render_template(
        'index.html',
        prediction=label,
        prediction_color=color,
        confidence=round(prob, 2),
        input_values=input_values
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)