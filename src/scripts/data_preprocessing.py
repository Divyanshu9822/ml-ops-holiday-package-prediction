import pandas as pd
from src.logger import logger
from src.entity.config_entity import DataPreprocessingConfig


class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def preprocess_and_save_data(self):
        df = pd.read_csv(self.config.raw_data_path)
        logger.info(f"Data loaded from {self.config.raw_data_path}")

        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
        df["MaritalStatus"] = df["MaritalStatus"].replace("Single", "Unmarried")

        df["Age"].fillna(df["Age"].median(), inplace=True)
        df["TypeofContact"].fillna(df["TypeofContact"].mode()[0], inplace=True)
        df["DurationOfPitch"].fillna(df["DurationOfPitch"].median(), inplace=True)
        df["NumberOfFollowups"].fillna(df["NumberOfFollowups"].mode()[0], inplace=True)
        df["PreferredPropertyStar"].fillna(
            df["PreferredPropertyStar"].mode()[0], inplace=True
        )
        df["NumberOfTrips"].fillna(0, inplace=True)
        df["NumberOfChildrenVisiting"].fillna(
            df["NumberOfChildrenVisiting"].mode()[0], inplace=True
        )
        df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)

        logger.info("Data preprocessed")

        df.to_csv(self.config.processed_data_path, index=False)
        logger.info(f"Data saved at {self.config.processed_data_path}")
