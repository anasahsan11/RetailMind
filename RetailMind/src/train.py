import json
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import MODEL_DIR, MODEL_PATH, METRICS_PATH, RANDOM_STATE, TEST_SIZE
from .data import load_dataset, inspect_dataset
from .pipeline import build_pipeline


def main():
    X, y, df = load_dataset()
    inspect_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index("Yes")]

    metrics = {
        "model": "LogisticRegression",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label="Yes", zero_division=0)),
        "roc_auc": float(roc_auc_score((y_test == "Yes").astype(int), probabilities)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\\n--- Day 1 evaluation ---")
    for name in ("accuracy", "precision", "recall", "roc_auc"):
        print(f"{name:>10}: {metrics[name]:.4f}")
    print(f"\\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
