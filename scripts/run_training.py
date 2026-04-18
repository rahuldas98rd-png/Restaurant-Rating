import os, sys, argparse, time

from src.logging.logger import get_logger
from src.exception.exception import CustomException
from src.pipeline.training_pipeline import TrainingPipeline
from src.constants.training_pipeline import PROJECT_NAME, PROJECT_VERSION


logging = get_logger(__name__)

class RunTraining:
    def __init__(self):
        self.project_name=PROJECT_NAME
        self.project_version=PROJECT_VERSION

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Train Regression Model")
        parser.add_argument("--config", default="src/entity/config_entity.py")
        return parser.parse_args()
    
    def training(self) -> None:
        try:
            start_time = time.time()
            args = RunTraining.parse_args()
            logging.info(f"Starting training pipeline — {self.project_name} v{self.project_version}")

            training_pipeline = TrainingPipeline()
            trained_artifact = training_pipeline.start_training()
            print(trained_artifact)

            elapsed = time.time() - start_time
            logging.info(f"\n{'='*60}")
            logging.info(f"Training complete in {elapsed:.1f}s")
            logging.info(f"{'='*60}")
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    run = RunTraining()
    run.training()
        

