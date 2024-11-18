import os,sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA

from src.exceptions import CustomException
from src.logger import logging
from src.utils import object_saver
from dataclasses import dataclass


@dataclass
class DataTransformationConfig():
    scaler_file_path=os.path.join('artifacts','scaler.pkl')
    pca_file_path=os.path.join('artifacts','pca.pkl')
    encoder_file_path=os.path.join('artifacts','encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def initiate_data_tranforming(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            x_cols=[['Area', 'Perimeter', 'MajorAxisLength','MinorAxisLength', 'AspectRation',
                'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 
                'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']]
            y_cols=['class']
            logging.info("Read train and test datasets intp Pandas DF")

            x_train_df=train_df.drop(columns=y_cols,axis=1)
            x_test_df=test_df.drop(columns=y_cols,axis=1)

            y_train=train_df.iloc[:,-1].values
            y_test=test_df.iloc[:,-1].values

            scaler=StandardScaler()
            encoder=LabelEncoder()
            pca=PCA(n_components=7)

            x_train_tranformed=pca.fit_transform(scaler.fit_transform(x_train_df))
            x_test_tranformed=pca.transform(scaler.transform(x_test_df))

            y_train_tranformed=encoder.fit_transform(y_train)
            y_test_tranformed=encoder.transform(y_test)

                
            logging.info("Using object_saver f'n to save neccessary preprocessing objects")
            object_saver(self.data_transformation_config.scaler_file_path,scaler)
            object_saver(self.data_transformation_config.pca_file_path,pca)
            object_saver(self.data_transformation_config.encoder_file_path,encoder)

            return (
                x_train_tranformed,y_train_tranformed,x_test_tranformed,y_test_tranformed,
                self.data_transformation_config.scaler_file_path,
                self.data_transformation_config.pca_file_path,
                self.data_transformation_config.encoder_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
