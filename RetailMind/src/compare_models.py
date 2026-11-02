
import json
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from sklearn.model_selection import train_test_split

from .config import MODEL_DIR, RANDOM_STATE, TEST_SIZE
from .data import load_dataset
from .pipeline import build_pipeline


def evaluate(name, estimator, X_train, X_test, y_train, y_test):

    model = build_pipeline(X_train, estimator)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    churn_index = list(model.classes_).index("Yes")

    probabilities = model.predict_proba(X_test)[:, churn_index]

    metrics = {
        "model": name,

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test, predictions,
                pos_label="Yes",
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test, predictions,
                pos_label="Yes",
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                (y_test == "Yes").astype(int),
                probabilities
            )
        )
    }

    return model, metrics


def main():

    X, y, _ = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = [

        ("Decision Tree", DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),

        ("Random Forest", RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )),

        ("SVM", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ]

    trained_models = {}
    results = []

    for name, estimator in models:

        print(f"Training {name}...")

        model, metrics = evaluate(
            name,
            estimator,
            X_train,
            X_test,
            y_train,
            y_test
        )

        trained_models[name] = model
        results.append(metrics)

    # Select champion using churn recall first.
    results.sort(
        key=lambda x: (x["recall"], x["roc_auc"]),
        reverse=True
    )

    champion_name = results[0]["model"]

    champion = trained_models[champion_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save champion model.
    joblib.dump(
        champion,
        MODEL_DIR / "churn_champion_day2.joblib"
    )

    # Save comparison results.
    comparison = {
        "selection_rule": "Highest churn recall; ROC-AUC breaks ties",
        "results": results,
        "champion": champion_name
    }

    with open(
        MODEL_DIR / "day2_model_comparison.json",
        "w"
    ) as file:

        json.dump(comparison, file, indent=4)

    print("\nMODEL COMPARISON")

    for result in results:

        print("\n", result["model"])

        for metric, value in result.items():

            if metric != "model":
                print(f"{metric}: {value:.4f}")

    print("\nCHAMPION:", champion_name)


if __name__ == "__main__":
    main()
