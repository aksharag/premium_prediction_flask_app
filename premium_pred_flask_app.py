from flask import Flask, request
import pickle
import joblib
import numpy as np

# master variable - controls entire application
app = Flask(__name__)

# model loading
model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

# API endpoints
@app.route('/')
def home():
    return "<h1>Insurance Cost Prediction</h1>"

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':        
        return "I will make the predictions."
    else:
        # post request along with the data
        # then i will make the prediction.
        req = request.get_json()
        age = req['age']
        height = req['height']
        weight = req['weight']
        bmi = weight/((height/100)**2)
        no_of_surgeries = req['surgeries']
        diabetes = req['diabetes']
        bp = req['bp']
        transplant = req['transplant']
        chronic_disease = req['chronic_disease']
        allergies = req['allergies']
        cancer = req['cancer']

        numerical_features = ['Age', 'Height', 'Weight', 'bmi']
        scaled_cols = scaler.transform(np.array([age,height,weight,bmi]).reshape(-1, 4))
        all_cols = np.concatenate((scaled_cols, diabetes, bp, transplant, chronic_disease, allergies, cancer, no_of_surgeries), axis=None).reshape(-1,11)

        pred = model.predict(all_cols)[0]
        return {"loan_approval_status":pred}