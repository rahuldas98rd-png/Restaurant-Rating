from src.exception.exception import CustomException
from src.logging.logger import get_logger
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import os, sys
from typing import List, Dict
from collections import Counter
from src.utils.main_utils.utils import (read_yaml_file, write_yaml_file, 
                                        read_csv, save_csv,
                                        save_numpy_array_data)

from src.entity.artifact_entity import (DataIngestionArtifact,
                                        PrimaryDataValidationArtifact,
                                        FinalDataValidationArtifact,
                                        DataTransformationArtifact)
from src.entity.config_entity import (PrimaryDataValidationConfig,
                                      DriftValidationConfig,
                                      FinalDataValidationConfig)
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
    def __init__(self, drift_validation_config:DriftValidationConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.drift_validation_config=drift_validation_config
            self.threshold=DATA_VALIDATION_DRIFT_THRESHOLD
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def drift_status(self, base_df:pd.DataFrame, current_df:pd.DataFrame) -> bool:
        try:
            # Assume data is HEALTHY (no drift) at the start
            status: bool = True # Variable to check overall drift status for the whole dataframe

            # Whether drift is found in THIS column
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
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def check_data_drift(self)->bool:
        try:
            base_df = read_csv(self.drift_validation_config.base_data_file_path)
            current_df = read_csv(self.data_transformation_artifact.transformed_train_file_path)
            status = self.drift_status(base_df=base_df, current_df=current_df)
            return status
        except Exception as e:
            raise CustomException(e, sys) from e
    


class FinalDataValidation:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                 data_validation_config:FinalDataValidationConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def column_check(self, dataframe:pd.DataFrame)->bool:
        try:
            final_schema_columns = self._schema_config['final_columns']
            if len(dataframe.columns) == len(final_schema_columns):
                if Counter(dataframe.columns) == Counter(final_schema_columns):
                    return True
                else:
                    logging.info(f"Require columns: {final_schema_columns}"\
                                 f"Dataframe columns: {dataframe.columns}")
                    return False
            else:
                logging.info(f"Required no. of columns: {len(final_schema_columns)}"\
                             f"Loaded no. of columns: {len(dataframe.columns)}")
                return False
            
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_final_data_validation(self) -> FinalDataValidationArtifact:
        try:
            logging.info("Initiate Final Data Validation Process.....")
            overall_status: List[bool] = []
            train_df = read_csv(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_df = read_csv(file_path=self.data_transformation_artifact.transformed_test_file_path)

            target_column = self._schema_config['data']['target_column']

            logging.info(f"Drop {target_column} from Train Dataset")
            X_train = train_df.drop(labels=target_column, axis=1)
            logging.info(X_train.head())
            y_train = train_df[target_column]

            logging.info(f"Drop {target_column} from Test Dataset")
            X_test = test_df.drop(labels=target_column, axis=1)
            logging.info(X_test.head())
            y_test = test_df[target_column]

            # Overall column check
            train_status = self.column_check(dataframe=X_train)
            overall_status.append(train_status)
            test_status = self.column_check(dataframe=X_test)
            overall_status.append(test_status)

            if all(overall_status):
                logging.info("Final validation successful.")
                X_train_arr = X_train.to_numpy()
                y_train_arr = np.array(y_train)
                X_test_arr = X_test.to_numpy()
                y_test_arr = np.array(y_test)

                train_arr = np.c_[X_train_arr, y_train_arr]
                test_arr = np.c_[X_test_arr, y_test_arr]

                save_numpy_array_data(file_path=self.data_validation_config.valid_train_file_path, array=train_arr)
                save_numpy_array_data(file_path=self.data_validation_config.valid_test_file_path, array=test_arr)

                data_validation_artifact=FinalDataValidationArtifact(
                    validation_status=all(overall_status),
                    valid_train_file_path=self.data_validation_config.valid_train_file_path,
                    valid_test_file_path=self.data_validation_config.valid_test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None
                )
                return data_validation_artifact
            else:
                logging.info("Final validation falied.")
                save_csv(file_path=self.data_validation_config.invalid_train_file_path, dataframe=train_df)
                save_csv(file_path=self.data_validation_config.invalid_test_file_path, dataframe=test_df)
                data_validation_artifact=FinalDataValidationArtifact(
                    validation_status=all(overall_status),
                    valid_train_file_path=None,
                    valid_test_file_path=None,
                    invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                    invalid_test_file_path=self.data_validation_config.invalid_test_file_path
                )
                return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e