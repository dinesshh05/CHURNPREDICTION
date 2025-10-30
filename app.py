import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 

app=Flask(__name__)
model=pickle.load(open('churn_prediction_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json
    df = pd.DataFrame([data])
    ordinal_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['Contract_ord'] = df['Contract'].map(ordinal_map)
    prediction = model.predict(df)[0]
    return jsonify({'prediction': int(prediction)})
    

if __name__=="__main__":
    app.run(debug=True)
