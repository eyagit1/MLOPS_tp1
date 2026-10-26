
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Load the dataset
df = pd.read_csv("churn_data.csv")
y = df["churn"]

ALL = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation",
]


def prepare(columns):
    X = pd.get_dummies(df[columns], drop_first=True)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


def run_trial(name, n_estimators, max_depth, columns):
    X_train, X_test, y_train, y_test = prepare(columns)

    with mlflow.start_run(run_name=name):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(columns))

        # Calculate and log metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(y_test, predictions, zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Save the trained model as an artifact
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )

        print(
            f"{name} -> accuracy: {accuracy:.3f} | "
            f"precision: {precision:.3f} | "
            f"recall: {recall:.3f} | f1: {f1:.3f}"
        )


# Group all trials in the same experiment
mlflow.set_experiment("churn-prediction")

# Trial 1: baseline
run_trial("trial-1", 100, 5, ALL)

# Trial 2: fewer trees and shallower trees
run_trial("trial-2", 50, 3, ALL)

# Trial 3: remove salary as a feature
run_trial(
    "trial-3",
    100,
    5,
    ["age", "abonnement", "anciennete", "consommation"],
)
