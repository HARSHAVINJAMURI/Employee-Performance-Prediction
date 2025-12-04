from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained model
with open(r"C:\Users\vinja\Downloads\FSDS\Gen AI\z1\gwp.pkl", "rb") as f:
    model = pickle.load(f)

# This list must match the training features (X.columns order BEFORE preprocessing)
# Example (adjust according to your dataset):
EXPECTED_FEATURES = [
    'quarter', 'department', 'day', 'team', 'targeted_productivity',
    'smv', 'wip', 'over_time', 'incentive',
    'idle_time', 'idle_men', 'no_of_style_change',
    'no_of_workers', 'year', 'month', 'dayofweek'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get form data
    # Assume you collect these fields in HTML form
    quarter = request.form['quarter']
    department = request.form['department']
    day = request.form['day']
    team = int(request.form['team'])
    targeted_productivity = float(request.form['targeted_productivity'])
    smv = float(request.form['smv'])
    wip = int(request.form['wip'])
    over_time = int(request.form['over_time'])
    incentive = int(request.form['incentive'])
    idle_time = float(request.form['idle_time'])
    idle_men = int(request.form['idle_men'])
    no_of_style_change = int(request.form['no_of_style_change'])
    no_of_workers = int(request.form['no_of_workers'])

    # date fields (for year, month, dayofweek)
    date_str = request.form['date']  # 'YYYY-MM-DD'
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    month = dt.month
    dayofweek = dt.weekday()

    # Create dataframe with single row in same format as training
    data = {
        'quarter': [quarter],
        'department': [department],
        'day': [day],
        'team': [team],
        'targeted_productivity': [targeted_productivity],
        'smv': [smv],
        'wip': [wip],
        'over_time': [over_time],
        'incentive': [incentive],
        'idle_time': [idle_time],
        'idle_men': [idle_men],
        'no_of_style_change': [no_of_style_change],
        'no_of_workers': [no_of_workers],
        'year': [year],
        'month': [month],
        'dayofweek': [dayofweek]
    }

    input_df = pd.DataFrame(data, columns=EXPECTED_FEATURES)

    # Predict
    prediction = model.predict(input_df)[0]

    return render_template('result.html',
                           prediction=round(prediction, 3),
                           model_name="gwp.pkl (best model)")

if __name__ == '__main__':
    app.run(debug=True)
