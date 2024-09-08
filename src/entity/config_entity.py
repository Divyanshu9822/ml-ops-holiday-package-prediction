from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    raw_data_path: Path
    processed_data_path: Path
    all_schema: dict
    STATUS_FILE: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    processed_data_path: Path
    preprocessor_model_name: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    trained_model_name: str
    preprocessor_model_path: Path
    model_name: str
    model_params: dict
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    preprocessor_model_path: Path
    model_path: Path
    model_name: str
    model_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
