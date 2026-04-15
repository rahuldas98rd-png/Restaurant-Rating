import sys, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.constants.training_pipeline import SCHEMA_FILE_PATH, ROBUST_SCALER_PARAMS

from src.entity.artifact_entity import PrimaryDataValidationArtifact,DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig

from src.logging.logger import get_logger
from src.exception.exception import CustomException
from src.utils.main_utils.utils import read_csv, save_object, read_yaml_file, save_csv

logging = get_logger(__name__)


class DataTransformation:
    def __init__(self, data_validation_artifact:PrimaryDataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def handle_missing_values(self, datframe:pd.DataFrame) -> pd.DataFrame:
        try:
            datframe.dropna(axis=0, subset=["Cuisines"], inplace=True)
            datframe.reset_index(drop=True, inplace=True)
            return datframe
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def binary_encoding(self, dataframe:pd.DataFrame) -> pd.DataFrame:
        try:
            binary_mapping = {'No':0, 'Yes':1}
            binary_columns = self._schema_config['binary_columns']
            for col in binary_columns:
                dataframe[col] = dataframe[col].replace(binary_mapping)
            return dataframe
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def target_driven_encoding(self, dataframe:pd.DataFrame, X:pd.DataFrame, target:str) -> pd.DataFrame:
        try:
            target_encoding_columns = self._schema_config['target_encoding_columns']
            for col in target_encoding_columns:
                df_rated = dataframe[dataframe[target]>0]
                city_avg_rating = (df_rated.groupby([col])[target]
                                .agg(average_rating='mean')
                                .sort_values(by='average_rating')
                                .reset_index())
                city_mapping = dict(zip(city_avg_rating[col], city_avg_rating['average_rating']))
                X[col] = X[col].replace(city_mapping)

            return X
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def feature_build(self, dataframe:pd.DataFrame)->pd.DataFrame:
        try:
            dataframe["Cuisine_Count"] = dataframe["Cuisines"].str.split(", ").apply(len)
            return dataframe
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def encode_cuisines(self, dataframe:pd.DataFrame, X:pd.DataFrame)->pd.DataFrame:
        try:
            dataframe = (
                dataframe.assign(Cuisine=dataframe["Cuisines"]
                                 .str.split(", "))
                                 .explode("Cuisine")
                        )
            dataframe["Cuisine"] = dataframe["Cuisine"].str.strip()

            stats = (
                dataframe.groupby("Cuisine")["Aggregate rating"]
                        .agg(Avg_Rating="mean")
                        .reset_index()
                    )
            cuisine_mapping = dict(zip(stats['Cuisine'], stats['Avg_Rating']))
            dataframe['Cuisine'] = dataframe['Cuisine'].replace(cuisine_mapping)

            cuisine_rating_train = (
                        dataframe.groupby("Restaurant ID")['Cuisine']
                        .agg(Cuisine_avg_rating='mean')
                        .reset_index()
                    )
            X = pd.merge(X, cuisine_rating_train, how='inner',on='Restaurant ID')
            return X
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
        
    def get_preprocessor(self) -> ColumnTransformer:
        """
        It initialises a RobustScaler object and returns 
        a ColumnTransformer object .

        Returns:
          A ColumnTransformer object
        """
        logging.info("Entered get_preprocessor method of DataTransformation class")
        try:
           transform_columns: list = self._schema_config['transformation_columns']
           scaler = RobustScaler(**ROBUST_SCALER_PARAMS)
           logging.info(f"Initialize RobustScaler object with {ROBUST_SCALER_PARAMS}")

           transformer_pipeline = Pipeline(steps=
                                            [
                                                ("scaler",scaler)
                                            ]
                                        )
           preprocessor = ColumnTransformer(transformers=[
               ("transformer",transformer_pipeline,transform_columns),
           ], remainder="passthrough")

           # by default preprocessor returns numpy array, to prevent that we need to specifically mention
           preprocessor.set_output(transform="pandas") 
           return preprocessor
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def drop_unwanted_columns(self, dataframe:pd.DataFrame)->pd.DataFrame:
        try:
            drop_labels = self._schema_config['drop_columns']
            dataframe = pd.DataFrame(dataframe)
            dataframe = dataframe.drop(labels=drop_labels, axis=1)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys) from e
        
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
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                train_df = read_csv(self.data_validation_artifact.valid_train_file_path)
                logging.info(f"Loaded train dataset: {train_df}")
                test_df = read_csv(self.data_validation_artifact.valid_test_file_path)
                logging.info(f"Loaded test dataset: {test_df}")

                target_column = self._schema_config['data']['target_column']

                # Missing value handling
                logging.info(f"{'='*15}Missing values in train dataset before{'='*15}")
                logging.info(train_df.isnull().sum())
                X_train = self.handle_missing_values(datframe=train_df)
                logging.info(f"{'='*15}Missing values in train dataset after{'='*15}")
                logging.info(train_df.isnull().sum())
                X_test = self.handle_missing_values(datframe=test_df)
                logging.info("Missing values handled from test dataset")

                # Binary Encoding
                logging.info("Perform binary encoding")
                X_train = self.binary_encoding(dataframe=X_train)
                logging.info(X_train.head())
                X_test = self.binary_encoding(dataframe=X_test)

                # Target Driven Encoding
                logging.info("Perform target driven encoding")
                X_train = self.target_driven_encoding(dataframe=train_df, X=X_train, target=target_column)
                logging.info(X_train.head())
                X_test = self.target_driven_encoding(dataframe=test_df, X=X_test, target=target_column)

                # Feature building
                logging.info("Perform new feature building")
                X_train = self.feature_build(dataframe=X_train)
                logging.info(X_train.head())
                X_test = self.feature_build(dataframe=X_test)

                # Encode cuisines
                logging.info("Perform cuisines encoding")
                X_train = self.encode_cuisines(dataframe=train_df, X=X_train)
                logging.info(X_train.head())
                X_test = self.encode_cuisines(dataframe=test_df, X=X_test)

                # Drop unwanted columns
                logging.info("Drop unwanted columnsdrop_unwanted_columns")
                X_train = self.drop_unwanted_columns(dataframe=X_train)
                logging.info(X_train.head())
                X_test = self.drop_unwanted_columns(dataframe=X_test)

                # Log transform
                transform_columns: list = self._schema_config['transformation_columns']
                logging.info(f"Perform log transformation on {transform_columns}")
                X_train = self.log_transform(dataframe=X_train)
                logging.info(X_train.head())
                X_test = self.log_transform(dataframe=X_test)

                logging.info(f"Perform scaling on {transform_columns}")
                preprocessor = self.get_preprocessor()
                logging.info("Applying fit_transform on Train Dataset")
                X_train = preprocessor.fit_transform(X_train)
                logging.info(X_train.head())
                logging.info("Applying transform on Test Dataset")
                X_test = preprocessor.transform(X_test)
                save_object(file_path="final_model/preprocessor.pkl", obj=preprocessor)
                
                logging.info(f"Rename columns {X_train.columns}")
                X_train = self.rename_columns(dataframe=X_train)
                logging.info(X_train.head())
                X_test = self.rename_columns(dataframe=X_test)
                
                save_csv(file_path=self.data_transformation_config.transformed_train_file_path,
                         dataframe=X_train)
                save_csv(file_path=self.data_transformation_config.transformed_test_file_path,
                         dataframe=X_test)
                save_object(file_path=self.data_transformation_config.transformed_object_file_path,
                            obj=preprocessor)
                
                data_transformation_artifact=DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                logging.info(f"Initialized data transformation artifact: {data_transformation_artifact}")
                return data_transformation_artifact

            else:
                logging.info("Valid data file not found!")
        except Exception as e:
            raise CustomException(e, sys) from e