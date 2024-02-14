import pickle
import pandas as pd
from flask import Response
import numpy as np
from flask import Flask,render_template,jsonify,request,app
from sklearn.preprocessing import StandardScaler
Application=Flask(__name__)
app=Application
scaler=pickle.load(open('models/diabetesscaler.pkl','rb'))
lr=pickle.load(open('models/diabetesmodel.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=lr.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')



if __name__=='__main__':
     app.run(host="0.0.0.0")
