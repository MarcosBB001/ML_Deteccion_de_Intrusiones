from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# DATASET PATHS
DATA_DIR = BASE_DIR / "datasets"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CIC_TRAIN_PATH = RAW_DIR / "cic_iot_2023_train.parquet"
CIC_TEST_PATH = RAW_DIR / "cic_iot_2023_test.parquet"

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"