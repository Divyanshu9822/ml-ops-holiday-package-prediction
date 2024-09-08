from src.logger import logger
from sklearn.model_selection import train_test_split
import joblib
from src.entity.config_entity import DataTransformationConfig
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.processed_data_path)

        train, test = train_test_split(data, test_size=0.30, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

    def fit_and_save_column_transformer(self):
        train = pd.read_csv(os.path.join(self.config.root_dir, "train.csv"))

        train_X = train.drop("ProdTaken", axis=1)

        cat_features = train_X.select_dtypes(include="object").columns
        num_features = train_X.select_dtypes(exclude="object").columns

        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder(drop="first")

        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, cat_features),
                ("StandardScaler", numeric_transformer, num_features),
            ]
        )

        preprocessor.fit(train_X)

        preprocessor_model_path = os.path.join(
            self.config.root_dir, self.config.preprocessor_model_name
        )
        joblib.dump(preprocessor, preprocessor_model_path)
        logger.info(f"Saved preprocessor model at {preprocessor_model_path}")
