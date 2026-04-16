import os, sys
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from src.exception.exception import CustomException
from src.logging.logger import get_logger
from src.utils.main_utils.utils import read_yaml_file, load_object
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from typing import List
import warnings
warnings.filterwarnings('ignore')


logging = get_logger(__name__)

class RatingPredictor:
    def __init__(self):
        self.file_dir = "final_model"
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def binary_encoding(self, dataframe:pd.DataFrame, binary_mapping:dict)->pd.DataFrame:
        try:
            binary_columns = self._schema_config['binary_columns']
            for col in binary_columns:
                dataframe[col] = dataframe[col].replace(binary_mapping)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def target_driven_encoding(self, dataframe:pd.DataFrame, target_mapping:List[dict])->pd.DataFrame:
        try:
            target_encoding_columns = self._schema_config['target_encoding_columns']
            for i,col in enumerate(target_encoding_columns):
                dataframe[col] = dataframe[col].replace(target_mapping[i])
            return dataframe
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def feature_build(self, dataframe:pd.DataFrame)->pd.DataFrame:
        try:
            dataframe["Cuisine_Count"] = dataframe["Cuisines"].str.split(", ").apply(len)
            return dataframe
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def encode_cuisines(self, dataframe:pd.DataFrame,cuisine_mapping:dict)->pd.DataFrame:
        try:
            dataframe_copy = dataframe
            dataframe_copy = (
            dataframe_copy.assign(Cuisine=dataframe_copy["Cuisines"]
                                .str.split(", "))
                                .explode("Cuisine")
                    )
            dataframe_copy["Cuisine"] = dataframe_copy["Cuisine"].str.strip()

            dataframe_copy['Cuisine'] = dataframe_copy['Cuisine'].replace(cuisine_mapping)

            if dataframe_copy['Cuisine'].dtypes=='object':
                dataframe_copy=dataframe_copy[~((dataframe_copy['Cuisine']=='B�_rek') | (dataframe_copy['Cuisine']=='Bihari'))]

            cuisine_rating = (
                        dataframe_copy.groupby("Restaurant ID")['Cuisine']
                        .agg(Cuisine_avg_rating='mean')
                        .reset_index()
                    )
            dataframe = pd.merge(dataframe, cuisine_rating, how='inner',on='Restaurant ID')
            return dataframe
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def log_transform(self, dataframe:pd.DataFrame)->pd.DataFrame:
        try:
            transform_columns: list = self._schema_config['transformation_columns']
            for col in transform_columns:
                dataframe[col] = np.log1p(dataframe[col].clip(lower=0))
            return dataframe
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def rename_columns(self, dataframe:pd.DataFrame)->pd.DataFrame:
        try:
            dataframe.columns = (
                    dataframe.columns
                    .str.replace(r'^(transformer__|remainder__)', '', regex=True)
                    .str.strip()
                )
            return dataframe
        except Exception as e:
            raise CustomException(e,sys) from e

    def transform(self, dataframe:pd.DataFrame):
        try:
            target_mapping: List[dict]

            preprocessor = load_object(file_path=os.path.join(self.file_dir,"preprocessor.pkl"))

            binary_mapping = read_yaml_file(file_path=os.path.join(self.file_dir,"binary_mapping.yaml"))
            city_mapping = read_yaml_file(file_path=os.path.join(self.file_dir,"City_mapping.yaml"))
            country_mapping = read_yaml_file(file_path=os.path.join(self.file_dir,"Country Code_mapping.yaml"))
            currency_mapping = read_yaml_file(file_path=os.path.join(self.file_dir,"Currency_mapping.yaml"))
            target_mapping = [city_mapping,currency_mapping,country_mapping]
            cuisine_mapping = read_yaml_file(file_path=os.path.join(self.file_dir,"cuisine_mapping.yaml"))

            # Binary Encoding
            logging.info("Perform binary encoding")
            dataframe = self.binary_encoding(dataframe=dataframe,binary_mapping=binary_mapping)
            logging.info(f'\n{dataframe.head()}')

            # Target Driven Encoding
            logging.info("Perform target driven encoding")
            dataframe = self.target_driven_encoding(dataframe=dataframe, target_mapping=target_mapping)
            if dataframe['City'].dtypes=='object':
                dataframe=dataframe[~((dataframe['City']=='Yorkton') | (dataframe['City']=='San Juan City'))]
            logging.info(f'\n{dataframe.head()}')

            # Feature building
            logging.info("Perform new feature building")
            dataframe = self.feature_build(dataframe=dataframe)
            logging.info(f'\n{dataframe.head()}')

            # Encode cuisines
            logging.info("Perform cuisines encoding")
            dataframe = self.encode_cuisines(dataframe=dataframe, cuisine_mapping=cuisine_mapping)
            dataframe = dataframe.drop(labels=['Cuisines'], axis=1)
            if 'Restaurant ID' in dataframe.columns:
                dataframe = dataframe.drop(labels=['Restaurant ID'], axis=1)
            logging.info(f'\n{dataframe.head()}')

            # Log transform
            logging.info(f"Perform log transformation")
            dataframe = self.log_transform(dataframe=dataframe)
            logging.info(f'\n{dataframe.head()}')

            # Normalization
            dataframe = preprocessor.transform(dataframe)
            logging.info(f'\n{dataframe.head()}')

            # Rename Columns
            logging.info(f"Rename columns {dataframe.columns}")
            dataframe = self.rename_columns(dataframe=dataframe)
            logging.info(f'\n{dataframe.head()}')
            
            return dataframe

        except Exception as e:
            raise CustomException(e,sys) from e
        
    def predict_batch(self, dataframe: pd.DataFrame):
        try:
            model = load_object(file_path=os.path.join(self.file_dir,"best_model.pkl"))
            print(dataframe.columns)
            prediction_batch = self.transform(dataframe=dataframe)
            print(prediction_batch.columns)

            logging.info(f'\n{prediction_batch.info()}')
            for col in prediction_batch.columns:
                prediction_batch[col] = prediction_batch[col].astype(float)
            logging.info(f'\n{prediction_batch.info()}')

            y_pred = model.predict(prediction_batch)
            predicted_data = pd.DataFrame(data=y_pred, columns=['Predicted_Rating'])
            output = pd.concat([dataframe,predicted_data], axis=1)
            return output
        except Exception as e:
            raise CustomException(e,sys) from e