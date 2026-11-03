
import joblib
import pandas as pd

from .config import MODEL_DIR


# Model paths for each development stage
MODEL_PATHS = {
    "day1": MODEL_DIR / "churn_pipeline.joblib",
    "day2": MODEL_DIR / "churn_champion_day2.joblib",
    "day3": MODEL_DIR / "churn_v3.joblib",
}


def predict(customer, model_version="day3"):

    # Validate model version
    if model_version not in MODEL_PATHS:
        raise ValueError(
            f"Invalid model version: {model_version}. "
            f"Available versions: {list(MODEL_PATHS.keys())}"
        )

    # Get model path
    model_path = MODEL_PATHS[model_version]

    # Check model availability
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Please train the model first."
        )

    # Load trained model
    model = joblib.load(model_path)

    # Check single or batch prediction
    is_batch = isinstance(customer, list)

    records = customer if is_batch else [customer]

    if not records:
        return []

    # Convert input into DataFrame
    frame = pd.DataFrame(records)

    # Convert TotalCharges to numeric
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"],
            errors="coerce"
        )

    # Predict customer churn
    predictions = model.predict(frame)

    # Extract probability for churn = Yes
    churn_index = list(model.classes_).index("Yes")

    probabilities = model.predict_proba(frame)[:, churn_index]

    # Prepare prediction response
    results = [
        {
            "prediction": str(label),
            "churn_probability": float(probability),
            "model_version": model_version
        }
        for label, probability in zip(
            predictions,
            probabilities
        )
    ]

    return results if is_batch else results[0]
