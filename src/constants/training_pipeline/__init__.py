import os
import sys
import numpy as np
import pandas as pd

"""
Some project related details
"""
PROJECT_NAME: str = "Restaurant Rating Prediction using Regression Model"
PROJECT_VERSION : str = "1.0.0"


"""
Defining common constant variables for training pipeline
"""
PIPELINE_NAME: str = "Restaurant_Rating"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "RestaurantDataset.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "best_model.pkl"


"""
Data ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DATABASE_NAME: str = "Restaurant_DB"
DATA_INGESTION_COLLECTION_NAME: str = "restaurant_details"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_RANDOM_SEED : int = 42


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MISSING_VALUE_THRESHOLD: float = 0.1
ROBUST_SCALER_PARAMS: dict = {
    'with_centering':True,          # subtract median
    'with_scaling':True,            # scale by IQR
    'quantile_range':(25.0, 75.0),  # standard IQR
    'unit_variance':False           # keep IQR-based scaling (not variance=1)
}


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"

PRIMARY_DATA_VALIDATION_VALID_DIR: str = "primary_validated"
PRIMARY_DATA_VALIDATION_INVALID_DIR: str = "primary_invalid"
PRIMARY_DATA_VALIDATION_TRAIN_FILE_PATH: str = "train.csv"
PRIMARY_DATA_VALIDATION_TEST_FILE_PATH: str = "test.csv"

DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DRIFT_THRESHOLD: float = 0.05
BASE_DATA_DIR: str = "historical_data"
BASE_DATA_FILE_PATH: str = "base_df.csv"


FINAL_DATA_VALIDATION_VALID_DIR: str = "final_validated"
FINAL_DATA_VALIDATION_INVALID_DIR: str = "final_invalid"
FINAL_DATA_VALIDATION_TRAIN_FILE_PATH: str = "train.npy"
FINAL_DATA_VALIDATION_TEST_FILE_PATH: str = "test.npy"


"""
Model Trainer ralated constant start with MODE_TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "best_model.pkl"
MODEL_TRAINER_EVALUATION_DIR: str = "model_evaluation"
MODEL_TRAINER_EVALUATION_REPORT: str = 'all_model_performance_report.yaml'
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05