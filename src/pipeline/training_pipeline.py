import sys, os

from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    PrimaryDataValidationConfig,
    DataTransformationConfig,
    DriftValidationConfig,
    FinalDataValidationConfig,
    ModelTrainerConfig
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    PrimaryDataValidationArtifact,
    DataTransformationArtifact,
    FinalDataValidationArtifact,
    ModelTrainerArtifact
)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import PrimaryDataValidation,DriftValidation,FinalDataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

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
        
    def start_primary_data_validation(self, data_ingestion_artifact:DataIngestionArtifact) -> PrimaryDataValidationArtifact:
        try:
            self.primary_data_validation_config=PrimaryDataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            primary_data_validation=PrimaryDataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                                          data_validation_config=self.primary_data_validation_config)
            primary_validation_artifact=primary_data_validation.initiate_primary_data_validation()
            return primary_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_data_transformation(self, primary_data_validation_artifact:PrimaryDataValidationArtifact) -> DataTransformationArtifact:
        try:
            self.data_transformation_config=DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation=DataTransformation(data_validation_artifact=primary_data_validation_artifact,
                                                   data_transformation_config=self.data_transformation_config)
            data_transformtion_artifact=data_transformation.initiate_data_transformation()
            return data_transformtion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_drift_validation(self, data_transformation_artifact:DataTransformationArtifact) -> bool:
        try:
            self.drift_validation_config=DriftValidationConfig(training_pipeline_config=self.training_pipeline_config)
            drift_validation=DriftValidation(drift_validation_config=self.drift_validation_config,
                                             data_transformation_artifact=data_transformation_artifact)
            check_status = drift_validation.check_data_drift()
            return check_status
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_final_data_validation(self, data_transformation_artifact:DataTransformationArtifact)->FinalDataValidationArtifact:
        try:
            self.final_data_validation_config=FinalDataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            final_data_validation=FinalDataValidation(data_transformation_artifact=data_transformation_artifact,
                                                      data_validation_config=self.final_data_validation_config)
            final_data_validation_artifact=final_data_validation.initiate_final_data_validation()
            return final_data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_model_trainer(self, final_data_validation_artifact:FinalDataValidationArtifact)->ModelTrainerArtifact:
        try:
            self.model_trainer_config=ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer=ModelTrainer(data_validation_artifact=final_data_validation_artifact,
                                       model_trainer_config=self.model_trainer_config)
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_training(self) -> ModelTrainerArtifact:
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            primary_data_validation_artifact=self.start_primary_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(primary_data_validation_artifact=primary_data_validation_artifact)
            
            status = self.start_drift_validation(data_transformation_artifact=data_transformation_artifact)
            print(status)
            if status:
                final_data_validation_artifact=self.start_final_data_validation(data_transformation_artifact=data_transformation_artifact)
                model_trainer_artifact=self.start_model_trainer(final_data_validation_artifact=final_data_validation_artifact)
            else:
                logging.info("Validation Failure")

            # (data_transformation_artifact=data_transformation_artifact)

            # self.sync_artifact_dir_to_s3()
            # self.sync_saved_model_dir_to_s3()

            # return model_trainer
        except Exception as e:
            raise CustomException(e, sys)