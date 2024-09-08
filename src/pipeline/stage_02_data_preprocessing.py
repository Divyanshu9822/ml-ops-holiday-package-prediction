from src.config.configuration import ConfigurationManager
from src.scripts.data_preprocessing import DataPreprocessing

STAGE_NAME = "Data Preprocessing stage"


class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.preprocess_and_save_data()
