import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.common import read_yaml


class PredictionPipeline:
    def __init__(self):
        schema = read_yaml(Path("schema.yaml"))
        self.target_column = schema["TARGET_COLUMN"]["name"]
        self.columns = [
            col for col in schema["COLUMNS"].keys() if col != self.target_column
        ]

        self.preprocessor = joblib.load(
            Path("artifacts/data_transformation/preprocessor.joblib")
        )
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))

    def preprocess(self, data: np.ndarray):
        data = pd.DataFrame(data, columns=self.columns)

        return self.preprocessor.transform(data)

    def predict(self, data: np.ndarray):
        transformed_data = self.preprocess(data)

        return self.model.predict(transformed_data)[0]
