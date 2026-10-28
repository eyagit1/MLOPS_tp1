
import mlflow

runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"],
)

columns = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.precision",
    "metrics.recall",
    "metrics.f1",
]

print(runs[columns].to_string(index=False))

# Display the best run among those with an F1 score
valid_runs = runs.dropna(subset=["metrics.f1"])

if valid_runs.empty:
    print("No runs with an F1 metric were found.")
else:
    best = valid_runs.iloc[0]
    print("\nBest run:", best["tags.mlflow.runName"])
    print("Run ID:", best["run_id"])
    print("F1-score:", best["metrics.f1"])
