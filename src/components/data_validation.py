from src.exception.exception import CustomException
from src.logging.logger import get_logger
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import os, sys
from typing import List, Dict
from collections import Counter
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file, read_csv, save_csv

from src.entity.artifact_entity import DataIngestionArtifact,PrimaryDataValidationArtifact
from src.entity.config_entity import PrimaryDataValidationConfig,DriftValidationConfig
from src.constants.training_pipeline import SCHEMA_FILE_PATH, DATA_VALIDATION_DRIFT_THRESHOLD

logging = get_logger(__name__)


class PrimaryDataValidation:
    """
    Primary validation is performed on ingested data to
    verify whether the ingested data meets required conditions before performing data transformation
    """
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:PrimaryDataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def validate_number_of_columns(self, dataframe:pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config['data']['columns'])
            logging.info(f"Loaded Dataframe has {dataframe.shape[0]:,} rows X {dataframe.shape[1]} columns")
            logging.info(f"Required number of columns: {number_of_columns}")
            
            if len(dataframe.columns) == number_of_columns:
                logging.info("Number of column validation successful.")
                return True
            else:
                logging.info("Number of column validation failed.")
                return False
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns: List[str]

            df_numerical_columns = [col for col in dataframe.columns if (dataframe[col].dtypes=='int64') or (dataframe[col].dtypes=='float64')]
            numerical_columns = self._schema_config['data']['numerical_columns']

            logging.info(f"Dataframe has numerical columns: {df_numerical_columns}")
            logging.info(f"Required numerical columns: {numerical_columns}")

            if Counter(df_numerical_columns) == Counter(numerical_columns):
                logging.info("Number of numerical column validation successful.")
                return True
            else:
                logging.info("Number of numerical column validation failed.")
                return False
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def validate_categorical_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            categorical_columns: List[str]

            df_categorical_columns = [col for col in dataframe.columns if (dataframe[col].dtypes=='object') or (dataframe[col].dtypes=='str')]
            categorical_columns = self._schema_config['data']['categorical_columns']

            logging.info(f"Dataframe has categorical columns: {df_categorical_columns}")
            logging.info(f"Required categorical columns: {categorical_columns}")

            if Counter(df_categorical_columns) == Counter(categorical_columns):
                logging.info("Number of categorical column validation successful.")
                return True
            else:
                logging.info("Number of categorical column validation failed.")
                return False
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_primary_data_validation(self) -> PrimaryDataValidationArtifact:
        try:
            logging.info("Initiate Primary Data Validation Process.....")
            overall_status: List[bool] = []

            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read data from train and test
            train_df = read_csv(file_path=train_file_path)
            test_df = read_csv(file_path=test_file_path)

            # Validate number of columns
            train_status = self.validate_number_of_columns(dataframe=train_df)
            overall_status.append(train_status)

            test_status = self.validate_number_of_columns(dataframe=test_df)
            overall_status.append(test_status)

            # Validate numerical columns
            train_status = self.validate_numerical_columns(dataframe=train_df)
            overall_status.append(train_status)

            test_status = self.validate_numerical_columns(dataframe=test_df)
            overall_status.append(test_status)

            # Validate categorical columns
            train_status = self.validate_categorical_columns(dataframe=train_df)
            overall_status.append(train_status)

            test_status = self.validate_categorical_columns(dataframe=test_df)
            overall_status.append(test_status)

            if all(overall_status):
                dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(dir_path, exist_ok=True)
                save_csv(file_path=self.data_validation_config.valid_train_file_path, dataframe=train_df)
                save_csv(file_path=self.data_validation_config.valid_test_file_path, dataframe=test_df)
                data_validation_artifact = PrimaryDataValidationArtifact(
                    validation_status=all(overall_status),
                    valid_train_file_path=self.data_validation_config.valid_train_file_path,
                    valid_test_file_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None
                )
                logging.info("Data Validation completed successfully.")
            else:
                dir_path = os.path.dirname(self.data_validation_config.invalid_train_file_path)
                os.makedirs(dir_path, exist_ok=True)
                save_csv(file_path=self.data_validation_config.invalid_train_file_path, dataframe=train_df)
                save_csv(file_path=self.data_validation_config.invalid_test_file_path, dataframe=test_df)
                data_validation_artifact = PrimaryDataValidationArtifact(
                    validation_status=all(overall_status),
                    valid_train_file_path=None,
                    valid_test_file_path=None,
                    invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path=self.data_validation_config.invalid_test_file_path
                )
                logging.info("Data Validation failed.")

            logging.info(f"Initialized DataValdationArtifact: {data_validation_artifact}")
            logging.info("="*60)
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e


class DriftValidation:
    def __init__(self, drift_validation_config:DriftValidationConfig):
        try:
            self.drift_validation_config=drift_validation_config
            self.threshold=DATA_VALIDATION_DRIFT_THRESHOLD
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def detect_dataset_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame) -> bool:
        try:
            status: bool = True # Variable to check overall drift status for the whole dataframe
            is_found: bool # Variable to map drift status of each numerical column
            report: Dict = {} # To store the drift report
            
            for column in base_df.columns:
                data_1=base_df[column]
                data_2=current_df[column]

                """
                checking whether both base_df and current_df have 
                same data distribution in that particular columns or not
                """
                is_same_dist = ks_2samp(data1=data_1, data2=data_2)

                if is_same_dist.pvalue >= self.threshold:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report.update({
                    column:{
                        "p_value":float(is_same_dist.pvalue),
                        "drift_status":is_found
                    }
                })
            drift_report_file_path = self.drift_validation_config.drift_report_file_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            logging.info(f"Data drift validation: {status}")

            return status
        
        # Validate data drift only for training dataset
        # num_train_df = train_df[self._schema_config['data']['numerical_columns']]
        # num_test_df = test_df[self._schema_config['data']['numerical_columns']]
        # drift_status = self.detect_dataset_drift(base_df=num_train_df, current_df=num_test_df)
        # overall_status.append(drift_status)

        except Exception as e:
            raise CustomException(e, sys) from e
    


class FinalDataValidation:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_final_data_validation(self) -> PrimaryDataValidationArtifact:
        try:
            logging.info("Initiate Final Data Validation Process.....")
            overall_status: List[bool] = []


        except Exception as e:
            raise CustomException(e, sys) from e