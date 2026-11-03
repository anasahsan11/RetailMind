
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve,
)

from .config import MODEL_DIR, RANDOM_STATE, TEST_SIZE
from .data import load_dataset
from .pipeline import build_pipeline


def diagnose_bias_variance(train_recall, validation_recall):

    gap = train_recall - validation_recall

    if train_recall < 0.65 and validation_recall < 0.65:
        status = "Possible high bias (underfitting)"

    elif gap > 0.15:
        status = "Possible high variance (overfitting)"

    else:
        status = "Reasonable generalization"

    return gap, status


def update_readme(champion_name, champion_params, results):

    readme_path = MODEL_DIR.parent / "README.md"

    section_start = "<!-- DAY3_RESULTS_START -->"
    section_end = "<!-- DAY3_RESULTS_END -->"

    lines = [
        section_start,
        "",
        "## Day 3 — Model Tuning and Bias-Variance Analysis",
        "",
        f"**Selected champion:** {champion_name}",
        "",
        "**Best hyperparameters:**",
        "",
        "```json",
        json.dumps(champion_params, indent=2),
        "```",
        "",
        "### Cross-validation comparison",
        "",
        "| Model | Train Recall | CV Recall | Recall Gap | CV ROC-AUC | Diagnosis |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for name, result in results.items():

        lines.append(
            f"| {name} "
            f"| {result['train_recall']:.4f} "
            f"| {result['cv_recall']:.4f} "
            f"| {result['recall_gap']:.4f} "
            f"| {result['cv_roc_auc']:.4f} "
            f"| {result['bias_variance_status']} |"
        )

    lines.extend([
        "",
        "### Final holdout evaluation",
        "",
        "The selected champion was evaluated on the untouched test set.",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ])

    for metric, value in results["_holdout_metrics"].items():
        lines.append(f"| {metric} | {value:.4f} |")

    lines.extend([
        "",
        "**Saved model:** `models/churn_v3.joblib`",
        "",
        "**Detailed results:** `models/day3_tuning_results.json`",
        "",
        section_end,
    ])

    new_section = "\n".join(lines)

    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    else:
        content = "# RetailMind AI\n"

    if section_start in content and section_end in content:
        before = content.split(section_start)[0]
        after = content.split(section_end, 1)[1]
        content = before + new_section + after
    else:
        content = content.rstrip() + "\n\n" + new_section + "\n"

    readme_path.write_text(content, encoding="utf-8")


def main():

    # 1. Load dataset
    X, y, df = load_dataset()

    print("\nDataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Recall scorer for churn = Yes
    recall_scorer = make_scorer(
        recall_score,
        pos_label="Yes",
        zero_division=0,
    )

    # 4. Cross-validation
    cv_strategy = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 5. Candidate models and parameter grids
    candidates = {

        "LogisticRegression": (
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__C": [0.1, 1, 10],
            },
        ),

        "DecisionTree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__max_depth": [4, 8, None],
                "classifier__min_samples_leaf": [5, 15],
            },
        ),

        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {
                "classifier__n_estimators": [150, 300],
                "classifier__max_depth": [None, 12],
                "classifier__min_samples_leaf": [2, 5],
            },
        ),

        "SVM": (
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__C": [0.5, 1, 5],
                "classifier__gamma": ["scale", "auto"],
            },
        ),
    }

    # 6. Multi-metric scoring
    scoring = {
        "recall": recall_scorer,
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
    }

    comparison_results = {}
    best_estimators = {}

    # 7. GridSearchCV for every candidate
    for model_name, (classifier, parameter_grid) in candidates.items():

        print("\n" + "=" * 55)
        print(f"Tuning {model_name}")
        print("=" * 55)

        pipeline = build_pipeline(X_train, classifier)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=scoring,
            refit="recall",
            cv=cv_strategy,
            n_jobs=-1,
            return_train_score=True,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        best_index = grid_search.best_index_
        cv_results = grid_search.cv_results_

        train_recall = float(
            cv_results["mean_train_recall"][best_index]
        )

        validation_recall = float(
            cv_results["mean_test_recall"][best_index]
        )

        recall_gap, diagnosis = diagnose_bias_variance(
            train_recall,
            validation_recall,
        )

        result = {
            "best_params": grid_search.best_params_,
            "train_recall": train_recall,
            "cv_recall": validation_recall,
            "recall_gap": float(recall_gap),
            "bias_variance_status": diagnosis,
            "cv_roc_auc": float(
                cv_results["mean_test_roc_auc"][best_index]
            ),
            "cv_accuracy": float(
                cv_results["mean_test_accuracy"][best_index]
            ),
        }

        comparison_results[model_name] = result
        best_estimators[model_name] = grid_search.best_estimator_

        print("Best parameters:", grid_search.best_params_)
        print(f"Training recall:   {train_recall:.4f}")
        print(f"Validation recall: {validation_recall:.4f}")
        print(f"Recall gap:        {recall_gap:.4f}")
        print(f"Diagnosis:         {diagnosis}")
        print(f"CV ROC-AUC:        {result['cv_roc_auc']:.4f}")

    # 8. Select champion using CV recall
    champion_name = max(
        comparison_results,
        key=lambda name: (
            comparison_results[name]["cv_recall"],
            comparison_results[name]["cv_roc_auc"],
            -max(0, comparison_results[name]["recall_gap"]),
        ),
    )

    champion = best_estimators[champion_name]

    print("\nSelected champion:", champion_name)

    # 9. Evaluate on untouched test set
    predictions = champion.predict(X_test)

    probabilities = champion.predict_proba(X_test)[
        :, list(champion.classes_).index("Yes")
    ]

    holdout_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                pos_label="Yes",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                pos_label="Yes",
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                (y_test == "Yes").astype(int),
                probabilities,
            )
        ),
    }

    print("\n--- Day 3 Holdout Evaluation ---")

    for metric, value in holdout_metrics.items():
        print(f"{metric:>12}: {value:.4f}")

    # 10. Generate learning curve
    print("\nGenerating learning curve...")

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=champion,
        X=X_train,
        y=y_train,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        cv=cv_strategy,
        scoring=recall_scorer,
        n_jobs=-1,
    )

    learning_curve_results = {
        "train_sizes": train_sizes.tolist(),
        "mean_train_recall": train_scores.mean(axis=1).tolist(),
        "mean_validation_recall": validation_scores.mean(axis=1).tolist(),
    }

    # 11. Save model and results
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "churn_v3.joblib"
    results_path = MODEL_DIR / "day3_tuning_results.json"

    joblib.dump(champion, model_path)

    final_results = {
        "day": 3,
        "champion_model": champion_name,
        "champion_parameters": comparison_results[champion_name][
            "best_params"
        ],
        "cross_validation": comparison_results,
        "holdout_metrics": holdout_metrics,
        "learning_curve": learning_curve_results,
        "cv_folds": 3,
        "selection_metric": "recall",
    }

    results_path.write_text(
        json.dumps(final_results, indent=2),
        encoding="utf-8",
    )

    # 12. Update README
    readme_results = dict(comparison_results)
    readme_results["_holdout_metrics"] = holdout_metrics

    update_readme(
        champion_name,
        comparison_results[champion_name]["best_params"],
        readme_results,
    )

    print("\n" + "=" * 55)
    print("DAY 3 COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print("Champion:", champion_name)
    print("Saved model:", model_path)
    print("Saved results:", results_path)
    print("Updated README.md")


if __name__ == "__main__":
    main()
