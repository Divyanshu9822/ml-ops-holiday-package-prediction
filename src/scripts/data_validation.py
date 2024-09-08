import pandas as pd
from src.logger import logger
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def preprocess_data(self):
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

        df.drop("CustomerID", axis=1, inplace=True)
        df["TotalVisiting"] = (
            df["NumberOfChildrenVisiting"] + df["NumberOfPersonVisiting"]
        )
        df.drop(
            ["NumberOfChildrenVisiting", "NumberOfPersonVisiting"], axis=1, inplace=True
        )

        logger.info("Data preprocessed")

        return df

    def save_data(self, df):
        df.to_csv(self.config.processed_data_path, index=False)
        logger.info(f"Data saved at {self.config.processed_data_path}")

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.processed_data_path)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e
