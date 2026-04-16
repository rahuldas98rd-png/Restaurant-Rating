import os, sys
from dotenv import load_dotenv
from src.logging.logger import get_logger
from src.exception.exception import CustomException
from src.utils.main_utils.utils import fetch_data_from_database, save_csv, read_yaml_file
import pandas as pd
from src.utils.ml_utils.model.estimator import RatingPredictor
from src.entity.config_entity import TrainingPipelineConfig
from src.constants.training_pipeline import SCHEMA_FILE_PATH

load_dotenv()
batch_collection = os.getenv("BATCH_COLLECTION_NAME")

logging = get_logger(__name__)

class BatchPredict:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
    @staticmethod
    def import_data(self)->pd.DataFrame:
        try:
            df = fetch_data_from_database(collection_name=batch_collection)
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_batch_prediction(self) -> None:
        try:
            df=BatchPredict.import_data(self)

            drop_columns = ["Restaurant Name","Address","Locality","Locality Verbose",
                            "Longitude","Latitude","Switch to order menu",
                            "Rating color","Rating text"]
            logging.info(f"Drop unwanted columnsdrop_unwanted_columns: {drop_columns}")
            df = df.drop(labels=drop_columns,axis=1)
            logging.info(df.head())

            # Missing value handling
            logging.info(f"{'='*15}Missing values in dataset before{'='*15}")
            logging.info(df.isnull().sum())
            df.dropna(how='any', inplace=True)
            logging.info(f"{'='*15}Missing values in dataset after{'='*15}")
            logging.info(df.isnull().sum())

            rating_predictor = RatingPredictor()

            output = rating_predictor.predict_batch(dataframe=df)

            save_file_path=os.path.join(self.training_pipeline_config.artifact_dir,"predicted_data","output.csv")
            save_csv(file_path=save_file_path, dataframe=output)
        except Exception as e:
            raise CustomException(e, sys) from e