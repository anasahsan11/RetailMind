
import joblib
import pandas as pd

from .config import MODEL_PATH

DAY2_MODEL_PATH = (
    MODEL_PATH.parent / "churn_champion_day2.joblib"
)


def predict(customer, model_version="day2"):

    if model_version == "day1":
        model_path = MODEL_PATH

    elif model_version == "day2":
        model_path = DAY2_MODEL_PATH

    else:
        raise ValueError("Invalid model version")

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Train the model first."
        )

    model = joblib.load(model_path)

    is_batch = isinstance(customer, list)

    records = customer if is_batch else [customer]

    if not records:
        return []

    frame = pd.DataFrame(records)

    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"],
            errors="coerce"
        )

    predictions = model.predict(frame)

    churn_index = list(model.classes_).index("Yes")

    probabilities = model.predict_proba(frame)[:, churn_index]

    results = [
        {
            "prediction": str(label),
            "churn_probability": float(probability)
        }
        for label, probability in zip(
            predictions,
            probabilities
        )
    ]

    return results if is_batch else results[0]
