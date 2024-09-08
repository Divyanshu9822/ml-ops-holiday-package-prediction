from src.config.configuration import ConfigurationManager
from src.scripts.data_transformation import DataTransformation


STAGE_NAME = "Data Transformation stage"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.train_test_spliting()
        data_transformation.fit_and_save_column_transformer()
