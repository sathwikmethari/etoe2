import os,sys
import pandas as pd

from src.exceptions import CustomException
from src.logger import logging
from src.utils import sampler_utils

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class Create_csv:
    train_data_path: str=os.path.join("artifacts","train_data.csv")
    test_data_path: str=os.path.join("artifacts","test_data.csv")
    raw_data_path: str=os.path.join("artifacts","raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.created_csvs=Create_csv()
    def Intiating_ingestion(self):
        logging.info("Started Data Ingestion")
        try:
            df=pd.read_excel('Jp_notebook\Dry_Bean_Dataset.xlsx')
            logging.info("Read the data into a Pandas DataFrame")
            #print(f'df before samp {len(df)}')

            df=sampler_utils(df)
            #print(f'df after samp {len(df)}')

            os.makedirs(os.path.dirname(self.created_csvs.train_data_path),exist_ok=True)
            df.to_csv(self.created_csvs.raw_data_path,index=False,header=True)

            logging.info("Initiating train_test_split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.created_csvs.train_data_path,index=False,header=True)
            test_set.to_csv(self.created_csvs.test_data_path,index=False,header=True)

            logging.info("Completed train_test_split")
            return(
                self.created_csvs.train_data_path,
                self.created_csvs.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    csvs=DataIngestion()
    train_data,test_data=csvs.Intiating_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_tranforming(train_data,test_data)