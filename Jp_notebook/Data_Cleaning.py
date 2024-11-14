import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA

def bounds(data):                     #Replaces Outliers with median

    data_num= data.select_dtypes(exclude='object')  #Numerical DataFrame
    num_columns=[x for x in data_num.columns]              #List containg all Numerical column names

    print('Columns list \n',num_columns)
    
    IQR={}                         #Creating empty dicts for interquartile ranges, lower bounds and upper bounds of Independent Variables
    l_bound={}
    u_bound={}
    
    for a in num_columns:
        IQR[a]=float(data[a].quantile(0.75))-float(data[a].quantile(0.25))

        l_bound[a]=float(data[a].quantile(0.25))-1.5*IQR[a]
 
        u_bound[a]=float(data[a].quantile(0.75))+1.5*IQR[a]
    
    print('\nInter_Quartile_Range Dict \n',IQR)
    print('\n Lower_Bound Dict \n',l_bound)
    print('\nUpper_Bound Dict \n',u_bound)
    
    
    median_dict=(data[num_columns].median()).to_dict()      #dict has medians of all independent varibles
    
    print('\nMedian Dict \n',median_dict)

    return l_bound, u_bound, median_dict, num_columns

def replacer_main(data,l_bound, u_bound, median_dict, num_columns):

    def replacer(row, l_bound, u_bound, median_dict, num_columns):
        for col in num_columns:
            if row[col] > u_bound[col] or row[col] < l_bound[col]:
                row[col] = median_dict[col]  # Replace with median if out of bounds
        return row
      
    df = data.apply(lambda row: replacer(row, l_bound, u_bound, median_dict, num_columns), axis=1)
    x=df.select_dtypes(exclude='O')
    y=df.iloc[:,-1].values
    x_cols=[i for i in x.columns]
    print(y,x_cols)
    return x,y,x_cols                                

def data_cleaner(x,y,x_cols):
    sm = SMOTE()                                            #Using SMOTE from imblearn to resample Imbalanced Data
    x_res, y_res = sm.fit_resample(x, y)

    x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.20) #Splitting into train and test after resampling
    scaler=StandardScaler()
    Encoder=LabelEncoder()

    x_train=scaler.fit_transform(x_train)                   #Scaling after splitting to avoid data leakage
    x_test=scaler.transform(x_test)

    y_train=Encoder.fit_transform(y_train)                  #Label Encoding categorical values
    y_test=Encoder.transform(y_test)

    '''
    pca=PCA()
    pca.fit(x_train)

    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_) #finding number of components required,
    explained_variance = pca.explained_variance_ratio_                          #  while getting maximum variance
    n_components_required = np.argmax(cumulative_explained_variance >= 0.95) + 1 
    print(n_components_required,type(x_res),type(y_res))

    THE n_components_required=7
    '''
    pca=PCA(n_components=7,random_state=42)             
    x_train_pca=pca.fit_transform(x_train)
    x_test_pca=pca.transform(x_test)
  
    return x_train_pca,x_test_pca,y_train,y_test,scaler,Encoder,pca #returning required objects   