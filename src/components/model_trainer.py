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

    def initate_model_training(self,x_train_tranformed,y_train_tranformed,x_test_tranformed,y_test_tranformed):
        try:
            logging.info("Splitting train_arr & test_arr into dependent and independent varibles")
            X_train,X_test=x_train_tranformed,x_test_tranformed
            y_train,y_test=y_train_tranformed,y_test_tranformed
            
            logging.info("Created a models dict for training")
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
            logging.info("Calling evaluate_model f'n from utils for training")
            model_report :dict=evaluate_model(X_train,y_train,X_test,y_test,models)

            best_score=max(sorted(model_report.values()))   #sorting based on test_accuracies

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_score)]
            
            best_model=models[best_model_name]
            
            logging.info("Saved best model as pkl file using object_saver f'n")
            print(f"model is of type{type(best_model)}")
            object_saver(
                self.model_trainer_config.trainer_file_path,best_model
            )

        except Exception as e:
            raise CustomException(e,sys)
        
        return model_report[best_model_name]         #returns training accuracy of best model


