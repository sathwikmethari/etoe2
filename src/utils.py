import os,sys
import pandas as pd
import numpy as np
import dill
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.exceptions import CustomException

def sampler_utils(data):
    data_num= data.select_dtypes(exclude='object')  #Numerical DataFrame
    num_columns=[x for x in data_num.columns]              #List containg all Numerical column names

    #print('Columns list \n',num_columns)
    
    IQR={}                         #Creating empty dicts for interquartile ranges, lower bounds and upper bounds of Independent Variables
    l_bound={}
    u_bound={}
    
    for a in num_columns:
        IQR[a]=float(data[a].quantile(0.75))-float(data[a].quantile(0.25))

        l_bound[a]=float(data[a].quantile(0.25))-1.5*IQR[a]
 
        u_bound[a]=float(data[a].quantile(0.75))+1.5*IQR[a]
    
    #print('\nInter_Quartile_Range Dict \n',IQR)
    #print('\n Lower_Bound Dict \n',l_bound)
    #print('\nUpper_Bound Dict \n',u_bound)
    
    
    median_dict=(data[num_columns].median()).to_dict()      #dict has medians of all independent varibles
    
    #print('\nMedian Dict \n',median_dict)
 
    def replacer(row, l_bound, u_bound, median_dict, num_columns):
        for col in num_columns:
            if row[col] > u_bound[col] or row[col] < l_bound[col]:
                row[col] = median_dict[col]  # Replace with median if out of bounds
        return row
      
    df = data.apply(lambda row: replacer(row, l_bound, u_bound, median_dict, num_columns), axis=1)
    x=df.select_dtypes(exclude='O')
    y=df.iloc[:,-1].values
    x_cols=list(x.columns)
    sm = SMOTE()                                            #Using SMOTE from imblearn to resample Imbalanced Data
    x_res, y_res = sm.fit_resample(x, y)
    x_res['class']=y_res
    return x_res                          

def object_saver(path,obj):
    try:
        dir_path=os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)

        with open(path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models_dict):
    try:
        report={}
        for i in range(len(list(models_dict))):
            model=list(models_dict.values())[i]
            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_acc=accuracy_score(y_train,y_train_pred)
            #train_f1=f1_score(y_train,y_train_pred,average='weighted')
            #train_prec=precision_score(y_train,y_train_pred,average='weighted')
            #train_recall=recall_score(y_train,y_train_pred,average='weighted')

            test_acc=accuracy_score(y_test,y_test_pred)
            #test_f1=f1_score(y_test,y_test_pred,average='weighted')
            #test_prec=precision_score(y_test,y_test_pred,average='weighted')
            #test_recall=recall_score(y_test,y_test_pred,average='weighted')

            report[list(models_dict.keys())[i]]=[test_acc]
    except Exception as e:
        raise CustomException(e,sys)
    return report