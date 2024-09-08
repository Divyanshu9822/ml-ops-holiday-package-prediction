from src.config.configuration import ConfigurationManager
from src.scripts.data_validation import DataValidation

STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        df = data_validation.preprocess_data()
        data_validation.save_data(df)
        data_validation.validate_all_columns()
