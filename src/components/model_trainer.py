import os, sys
from typing import Dict, List
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from src.exception.exception import CustomException
from src.logging.logger import get_logger
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import FinalDataValidationArtifact,ModelTrainerArtifact,RegressionMetricArtifact

from src.utils.main_utils.utils import save_object,load_numpy_array_data
from src.utils.ml_utils.metric.regression_metric import get_regression_score, evaluate_models
from src.constants.models import MODELS, PARAMETERS
import mlflow

logging = get_logger(__name__)

class ModelTrainer:
    def __init__(self,data_validation_artifact:FinalDataValidationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_validation_artifact = data_validation_artifact
            self.__models=MODELS
            self.__model_params=PARAMETERS
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def track_mlflow(self, best_model, regression_metric:RegressionMetricArtifact)->None:
        try:
            with mlflow.start_run():
                r2_score = regression_metric.r2_score
                mae = regression_metric.mean_absolute_error
                rmse = regression_metric.root_mean_squared_error

                mlflow.log_metric("r2_score", r2_score)
                mlflow.log_metric("mean_absolute_error", mae)
                mlflow.log_metric("root_mean_squared_error", rmse)
                mlflow.sklearn.log_model(best_model, 'best_model')
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def train_model(self,X_train: np.array, y_train: np.array,X_test: np.array, y_test: np.array,
                    models:dict,models_report:dict):
        try:
            # To get best model accuracy from dict
            r2_score_list: List = []
            model_keys_list: List = []

            for key in models_report.keys():
                r2_score_list.append(models_report[key]['model_test_r2_score'])
                model_keys_list.append(key)

            model_score:dict = dict(zip(model_keys_list, r2_score_list))
            best_model_score:float = max(r2_score_list)

            # To get best model name from dict
            best_model_name:str = list(model_score.keys())[list(model_score.values()).index(best_model_score)]
            best_model = models[best_model_name]
            best_model_params = models_report[best_model_name]['best_params']
            best_model.set_params(**best_model_params)
            best_model.fit(X_train,y_train)

            y_train_pred=best_model.predict(X_train)
            regression_train_metric=get_regression_score(y_true=y_train,y_pred=y_train_pred)
            self.track_mlflow(best_model, regression_train_metric) # Track the experiments with mlflow

            y_test_pred=best_model.predict(X_test)
            regression_test_metric=get_regression_score(y_true=y_test,y_pred=y_test_pred)
            self.track_mlflow(best_model, regression_test_metric) # Track the experiments with mlflow
            return (best_model,best_model_name,best_model_score,\
                    regression_train_metric,regression_test_metric)

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(file_path=self.data_validation_artifact.valid_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_validation_artifact.valid_test_file_path)

            logging.info("Separating X_train, y_train X_test, y_test")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models_report: Dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=self.__models,params=self.__model_params)
            
            (best_model,best_model_name,best_model_score,\
            regression_train_metric,regression_test_metric)=self.train_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                                        models=self.__models,models_report=models_report)
            


            logging.info(f"Saving best model: {best_model_name} with accuracy score: {best_model_score}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            save_object(file_path="final_model/best_model.pkl", obj=best_model)
            
            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        train_metric_artifact=regression_train_metric,
                                                        test_metric_artifact=regression_test_metric)

            logging.info(f"Sucessfully initialized ModelTrainerArtifact: {model_trainer_artifact}")
            logging.info("Model trained successfully.")
            logging.info("="*60)
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)