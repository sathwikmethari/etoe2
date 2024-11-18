import pandas as pd
import numpy as np
from flask import Flask,request,render_template
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.pipeline.predict_pipeline import prediction_pipiline


application=Flask(__name__)
app=application
@app.route('/')
def home():
    return render_template('home.html',url2='http://localhost:5000/predict')

@app.route('/predict' ,methods=['GET','POST'])
def predict():
    output='Result'
    input_dict={}
    errors=[]
    x_columns=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 
               'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 
               'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']
               
    if request.method=='POST':
        for column in x_columns:
            try:
                input_dict[column]=float(request.form.get(column))
            
            except Exception as e:
                errors.append(f"{column} ,{e} ")
            
        input_df=pd.DataFrame([input_dict])
        predict_pipeline=prediction_pipiline()
        output=predict_pipeline.predict_output(input_df)


    return render_template('predict.html', errors=errors, input_list=input_dict, columns=x_columns,\
                            l=len(x_columns),output=output[0])

if __name__=='__main__':
    app.run(host="0.0.0.0")