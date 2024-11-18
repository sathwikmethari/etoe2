import sys
import pandas as pd
from src.exceptions import CustomException
from src.utils import object_loader

class prediction_pipiline:
    def __init__(self):
        pass
    def predict_output(self,features):
        try:
            model_path='artifacts\model.pkl'
            scaler_path='artifacts\scaler.pkl'
            encoder_path='artifacts\encoder.pkl'
            pca_path='artifacts\pca.pkl'

            model1=object_loader(model_path)
            scaler=object_loader(scaler_path)
            encoder=object_loader(encoder_path)
            pca=object_loader(pca_path)
            print(type(model1))

            data_transformed= pca.transform(scaler.transform(features))
            prediction= encoder.inverse_transform(model1.predict(data_transformed))

            return prediction
        except Exception as e:
            raise CustomException(e,sys)
'''
class customData:
    def __init__(self,
    Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, 
    Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, 
    Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4):
        
        self.Area, self.Perimeter, self.MajorAxisLength = Area, Perimeter, MajorAxisLength
        self.MinorAxisLength, self.AspectRation, self.Eccentricity = MinorAxisLength, AspectRation, Eccentricity
        self.ConvexArea, self.EquivDiameter, self.Extent, self.Solidity = ConvexArea, EquivDiameter, Extent, Solidity
        self.roundness, self.Compactness, self.ShapeFactor1 = roundness, Compactness, ShapeFactor1
        self.ShapeFactor2, self.ShapeFactor3, self.ShapeFactor4 = ShapeFactor2, ShapeFactor3, ShapeFactor4


    def get_input_as_df(self):
        try:
            custom_input_dict={"Area":[self.Area], "Perimeter":[self.Perimeter], "MajorAxisLength":[self.MajorAxisLength],
            "MinorAxisLength":[self.MinorAxisLength], "AspectRation":[self.AspectRation], "Eccentricity":[self.Eccentricity],
            "ConvexArea":[self.ConvexArea], "EquivDiameter":[self.EquivDiameter], "Extent":[self.Extent], "Solidity":[self.Solidity],
            "roundness":[self.roundness], "Compactness":[self.Compactness], "ShapeFactor1":[self.ShapeFactor1],
            "ShapeFactor2":[self.ShapeFactor2], "ShapeFactor3":[self.ShapeFactor3], "ShapeFactor4":[self.ShapeFactor4]}
            
            return pd.DataFrame(custom_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
'''