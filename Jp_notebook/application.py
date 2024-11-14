import pandas as pd
import numpy as np
from flask import Flask,request,render_template
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data=pd.read_excel('Dry_Bean_Dataset.xlsx')         #Reading the dataset

from Data_Cleaning import replacer_main,data_cleaner          #Importing a function from .py file       


#Calling imported functions for preprocessing
#This f'n replaces outliers with median and returns updated x,y and xcolumn names
x,y,x_columns=replacer_main(data)
#This f'n resamples the data, scales and encodes the data, applies Dimensionality reduction(PCA) on it.                                  
x_train,x_test,y_train,y_test,Scaler,Encoder,Pca=data_cleaner(x,y,x_columns)    

#Chose XGB as it gave better metric scores
#Model parameters were found with help of RandomizedSearchCV

model=XGBClassifier(colsample_bytree=np.float64(0.8738286191519333), gamma=np.float64(0.043934055621328405),\
                learning_rate= np.float64(0.05164743203260611), max_depth=11, min_child_weight=1,\
                n_estimators= 314, reg_alpha= np.float64(0.5956387406078443), reg_lambda= np.float64(0.5715761885501583),\
                subsample= np.float64(0.7647363656589075))

model=XGBClassifier(colsample_bytree=np.float64(0.8950004992438994), gamma=np.float64(0.22610897044490358),\
                learning_rate= np.float64(0.07738144688199458), max_depth=14, min_child_weight=3,\
                n_estimators= 231, reg_alpha= np.float64(0.1763869865062233), reg_lambda= np.float64(0.5983677727394797),\
                subsample= np.float64(0.7675701798018192))
model.fit(x_train,y_train)

Input_dict=dict()                                        #Empty Dictionary,that we use to get inputs from html page
for column in x_columns:
    Input_dict[column]=0
output=['Result']

application=Flask(__name__)
app=application
@app.route('/')
def home():
    return render_template('home.html',url2='http://localhost:5000/predict')

@app.route('/predict' ,methods=['GET','POST'])
def predict():
    errors=[]
    if request.method=='POST':
        for column in x_columns:
            try:
                Input_dict[column]=float(request.form.get(column))
            
            except ValueError:
                errors.append(f"{column} , ")

        #Turning inputs into an array
        Input_array=np.array([input for input in Input_dict.values()],dtype=float)

        #Scaling and applying pca
        Input_array=Pca.transform(Scaler.transform([Input_array]))
        #print(Input_array)
        output[0]=Encoder.inverse_transform(model.predict(Input_array))[0]


    return render_template('predict.html', errors=errors, input_list=Input_dict, columns=x_columns,\
                            l=len(x_columns),output=output[0])

if __name__=='__main__':
    app.run(debug=True)

