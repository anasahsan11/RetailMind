import joblib
import pandas as pd
from .config import MODEL_PATH


def _load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def predict(customer, model_path=MODEL_PATH):
    """
    Predict churn for one customer dictionary or a list of dictionaries.

    Returns a dictionary for one customer, or a list of dictionaries for a batch.
    Each result contains the predicted Churn label and probability of 'Yes'.
    """
    model = _load_model(model_path)
    is_batch = isinstance(customer, list)
    records = customer if is_batch else [customer]

    if not records:
        return []

    frame = pd.DataFrame(records)
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")

    labels = model.predict(frame)
    yes_index = list(model.classes_).index("Yes")
    probabilities = model.predict_proba(frame)[:, yes_index]

    results = [
        {
            "prediction": str(label),
            "churn_probability": float(probability),
        }
        for label, probability in zip(labels, probabilities)
    ]
    return results if is_batch else results[0]
