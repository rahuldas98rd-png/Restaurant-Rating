import os, sys
from dotenv import load_dotenv
from src.logging.logger import get_logger
from src.exception.exception import CustomException
from src.utils.main_utils.utils import insert_data_into_database

logging = get_logger(__name__)

load_dotenv()

data_file_path = os.getenv("DATA_FILE_PATH")
data_collection = os.getenv("DATA_COLLECTION_NAME")

batch_file_path = os.getenv("BATCH_FILE_PATH")
batch_collection = os.getenv("BATCH_COLLECTION_NAME")

historical_file_path = os.getenv("HISTORICAL_DATA_FILE_PATH")
historical_collection = os.getenv("HISTORICAL_COLLECTION_NAME")

if __name__=="__main__":
    try:
        # Exporting main data file to MongoDB
        insert_data_into_database(collection_name=data_collection,
                                  file_path=data_file_path)

        # Exporting Test data file to MongoDB
        insert_data_into_database(collection_name=batch_collection,
                                  file_path=batch_file_path)
        
        # Exporting historical data to MongoDB
        insert_data_into_database(collection_name=historical_collection,
                                  file_path=historical_file_path)
    except Exception as e:
        raise CustomException(e, sys) from e