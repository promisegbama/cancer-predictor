from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
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

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

COMMENTS_FILE = 'comments.json'
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password123'  # Change to something secure

def load_comments():
    try:
        with open(COMMENTS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_comments(comments):
    with open(COMMENTS_FILE, 'w') as f:
        json.dump(comments, f)

@app.route('/')
def home():
    page = int(request.args.get('page', 1))
    comments = load_comments()
    per_page = 5
    start = (page - 1) * per_page
    end = start + per_page
    paginated_comments = comments[start:end]
    total_pages = (len(comments) + per_page - 1) // per_page

    return render_template("index.html",
                           comments=paginated_comments,
                           is_admin=session.get('is_admin', False),
                           current_page=page,
                           total_pages=total_pages)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('home'))
        return 'Invalid credentials'
    return '''
        <form method="post">
            <input name="username" placeholder="Username">
            <input name="password" type="password" placeholder="Password">
            <button type="submit">Login</button>
        </form>
    '''

@app.route('/logout')
def logout():
    session.pop('is_admin', None)
    return redirect(url_for('home'))

@app.route('/comment', methods=['POST'])
def comment():
    username = request.form.get('username') or 'Anonymous'
    feedback = request.form.get('feedback')
    ip = request.remote_addr
    device = request.headers.get('User-Agent')

    # 🔍 Capture user IP and browser info
    user_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent')
    print(f"📝 Feedback submitted by {username} from IP: {user_ip}, using: {user_agent}")

    if feedback:
        comments = load_comments()
        comments.append({
            'username': username,
            'text': feedback,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ip': ip,
            'device': device
        })
        save_comments(comments)

    return redirect(url_for('home'))


@app.route('/delete-comment/<int:comment_id>', methods=['POST'])
def delete_comment(comment_id):
    if not session.get('is_admin'):
        return 'Unauthorized', 403
    comments = load_comments()
    if 0 <= comment_id < len(comments):
        del comments[comment_id]
        save_comments(comments)
    return redirect(url_for('home'))


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