import sys, os

from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
)

from src.components.data_ingestion import DataIngestion

from src.logging.logger import get_logger
from src.exception.exception import CustomException


logging = get_logger(__name__)

class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_training(self): # -> returns ModelTrainerArtifact
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            # data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            # data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            # model_trainer=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            # self.sync_artifact_dir_to_s3()
            # self.sync_saved_model_dir_to_s3()

            # return model_trainer
        except Exception as e:
            raise CustomException(e, sys)