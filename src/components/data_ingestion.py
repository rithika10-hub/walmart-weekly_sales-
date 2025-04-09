import os
import pandas as pd
from src.logger.logging import logging
from src.exception.exception import CustomException
import sys

class DataIngestion:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            logging.info(f"📥 Attempting to load data from: {self.file_path}")
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            df = pd.read_csv(self.file_path)
            logging.info(f"✅ Data loaded successfully. Shape: {df.shape}")
            return df

        except Exception as e:
            logging.error("❌ Exception occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        data_path = os.path.join("artifacts", "cleaned_walmart_data.csv")
        ingestion = DataIngestion(file_path=data_path)
        df = ingestion.load_data()
    except Exception as e:
        logging.error("❌ Failed to run data ingestion script.")
        raise CustomException(e, sys)
