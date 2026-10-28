
import mlflow
import mlflow.sklearn
import pandas as pd

# Retrieve runs with a valid F1 score
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"],
)

runs = runs.dropna(subset=["metrics.f1"])

if runs.empty:
    raise RuntimeError("No run with an F1 score was found.")

# Select the best run and retrieve its saved model
best = runs.iloc[0]
run_id = best["run_id"]

model = mlflow.sklearn.load_model(
    f"runs:/{run_id}/model"
)

print("Selected run:", best["tags.mlflow.runName"])
print("Model reloaded:", type(model).__name__)
print("Expected features:", model.n_features_in_)

# Example customer
customer = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])

# Match the input columns and their order to the saved model
customer = customer.reindex(
    columns=model.feature_names_in_,
    fill_value=False,
)

prediction = model.predict(customer)[0]

print(
    "Will the customer leave?",
    "YES" if prediction == 1 else "NO",
)
