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


# """
# Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
# """
# DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
# DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
# DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
# PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

# ## KNN-imputer to replace nan values
# DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
#     "missing_values": np.nan,
#     "n_neighbors": 3,
#     "weights": "uniform",
# }


# """
# Data Validation related constant start with DATA_VALIDATION VAR NAME
# """
# DATA_VALIDATION_DIR_NAME: str = "data_validation"
# DATA_VALIDATION_VALID_DIR: str = "validated"
# DATA_VALIDATION_INVALID_DIR: str = "invalid"
# DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
# DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
# DATA_VALIDATION_DRIFT_THRESHOLD: float = 0.05
# DATA_VALIDATION_TRAIN_FILE_PATH: str = "train.npy"
# DATA_VALIDATION_TEST_FILE_PATH: str = "test.npy"
