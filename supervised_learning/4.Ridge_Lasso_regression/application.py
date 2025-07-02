from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler    
import sklearn


app = Flask(__name__, template_folder='templates')  # <-- Make sure this is set

print("Flask app is starting...")

# Load models
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
scaler_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index123.html')  # This will now find the HTML correctly

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=scaler_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])


    else:
       return   render_template('home.html')
 

if __name__ == '__main__':
    app.run(debug=True)