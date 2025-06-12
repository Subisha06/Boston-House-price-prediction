from flask import Flask, request, render_template
import pickle

import numpy as np

app = Flask(__name__)

# Load the model
with open("BostonHousing1.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['crim']),
        float(request.form['zn']),
        float(request.form['indus']),
        float(request.form['chas']),
        float(request.form['nox']),
        float(request.form['rm']),
        float(request.form['age']),
        float(request.form['dis']),
        float(request.form['rad']),
        float(request.form['tax']),
        float(request.form['ptratio'])
    ]

    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Predicted value: {prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
