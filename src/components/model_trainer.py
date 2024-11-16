import pandas as pd
import numpy as np
import os,sys
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.logger import logging
from src.exceptions import CustomException
from src.utils import object_saver,evaluate_model

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trainer_file_path=os.path.join("artifacts",'model.pkl')
class Model_trainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initate_model_training(self,train_arr,test_arr):
        try:
            logging.info("")
            X_train,X_test=train_arr[:,:-1],test_arr[:,:-1]
            y_train,y_test=train_arr[:,-1],test_arr[:,-1]
            
            models={
                "LogisticRegression":LogisticRegression(),
                "SupportVector":SVC(),
                "NaiveBayes":GaussianNB(),
                "KnnClassifier":KNeighborsClassifier(),
                "DecisionTree":DecisionTreeClassifier(),
                "RandomForest":RandomForestClassifier(),
                #"AdaBoost":AdaBoostClassifier(),    raises some warning, So, commented it out(has lower accuracy anyway)
                "GradientBoost":GradientBoostingClassifier(),
                "XgbClassifier":XGBClassifier()
            }

            model_report :dict=evaluate_model(X_train,y_train,X_test,y_test,models)

            best_score=max(sorted(model_report.values()))

            best_model=list(model_report.keys())[
                list(model_report.values()).index(best_score)]
            
            object_saver(
                self.model_trainer_config.trainer_file_path,best_model
            )

        except Exception as e:
            raise CustomException(e,sys)
        
        return model_report[best_model]         #returns training accuracy of best model


