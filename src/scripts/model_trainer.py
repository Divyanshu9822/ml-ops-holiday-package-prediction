from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from src.entity.config_entity import ModelTrainerConfig
import joblib
import pandas as pd
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        models = {
            "Logistic Regression": LogisticRegression,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "Gradient Boost": GradientBoostingClassifier,
            "Adaboost": AdaBoostClassifier,
            "Xgboost": XGBClassifier,
            "SVC": SVC,
            "Gaussian Naive Bayes": GaussianNB,
        }

        train_data = pd.read_csv(self.config.train_data_path)
        X_train = train_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]

        preprocessor = joblib.load(self.config.preprocessor_model_path)
        X_train = preprocessor.transform(X_train)

        model_name = self.config.model_name
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' is not supported.")

        model_class = models[model_name]
        model_params = self.config.model_params

        prod_model = model_class(**model_params)

        prod_model.fit(X_train, y_train)

        joblib.dump(
            prod_model,
            os.path.join(self.config.root_dir, self.config.trained_model_name),
        )
