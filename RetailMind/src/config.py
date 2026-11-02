from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "telco_customer_churn.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET_COLUMN = "Churn"
ID_COLUMNS = ["customerID"]
TEST_SIZE = 0.20
RANDOM_STATE = 42
