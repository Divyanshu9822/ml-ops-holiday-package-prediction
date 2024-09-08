import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import joblib
import xgboost
from src.entity.config_entity import ModelEvaluationConfig
from src.constants import *
from src.utils.common import save_json
import dagshub


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, y_test, y_test_pred):
        accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average="weighted")
        precision = precision_score(y_test, y_test_pred, average="weighted")
        recall = recall_score(y_test, y_test_pred, average="weighted")
        roc_auc = roc_auc_score(y_test, y_test_pred)
        return accuracy, f1, precision, recall, roc_auc

    def log_into_mlflow(self):
        dagshub.init(
            repo_owner="divyanshu9822",
            repo_name="ml-ops-holiday-package-prediction",
            mlflow=True,
        )
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        preprosessor = joblib.load(self.config.preprocessor_model_path)
        test_x_transformed = preprosessor.transform(test_x)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            pred_y = model.predict(test_x_transformed)

            accuracy, f1, precision, recall, roc_auc = self.eval_metrics(test_y, pred_y)

            scores = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.model_params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)

            if tracking_url_type_store != "file":
                if isinstance(model, xgboost.XGBModel):
                    mlflow.xgboost.log_model(
                        model, "model", registered_model_name=self.config.model_name
                    )
                else:
                    mlflow.sklearn.log_model(
                        model, "model", registered_model_name=self.config.model_name
                    )
            else:
                if isinstance(model, xgboost.XGBModel):
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
