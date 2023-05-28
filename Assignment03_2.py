import mlflow

# Track parameters
mlflow.log_param("max_depth", 20)

# Track metrics
mlflow.log_metric("rmse", 0.5)

# Track artifact
mlflow.log_artifact("/path/graph.png", "myGraph")

# Track model (depends on model type)
mlflow.sklearn.log_model(model, "myModel")
mlflow.keras.log_model(model, "myModel")
mlflow.pytorch.log_model(model, "myModel")



from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

EXPERIMENT_NAME = "mlflow-demo"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

for idx, depth in enumerate([1, 2, 5, 10, 20]):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Start MLflow
    RUN_NAME = f"run_{idx}"
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME) as run:
        # Retrieve run id
        RUN_ID = run.info.run_id

        # Track parameters
        mlflow.log_param("depth", depth)

        # Track metrics
        mlflow.log_metric("accuracy", accuracy)

        # Track model
        mlflow.sklearn.log_model(clf, "classifier")



import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "mlflow-demo"

client = MlflowClient()

# Retrieve Experiment information
EXPERIMENT_ID = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

# Retrieve Runs information (parameter 'depth', metric 'accuracy')
ALL_RUNS_INFO = client.list_run_infos(EXPERIMENT_ID)
ALL_RUNS_ID = [run.run_id for run in ALL_RUNS_INFO]
ALL_PARAM = [client.get_run(run_id).data.params["depth"] for run_id in ALL_RUNS_ID]
ALL_METRIC = [client.get_run(run_id).data.metrics["accuracy"] for run_id in ALL_RUNS_ID]

# View Runs information
run_data = pd.DataFrame({"Run ID": ALL_RUNS_ID, "Params": ALL_PARAM, "Metrics": ALL_METRIC})

# Retrieve Artifact from best run
best_run_id = run_data.sort_values("Metrics", ascending=False).iloc[0]["Run ID"]
best_model_path = client.download_artifacts(best_run_id, "classifier")
best_model = mlflow.sklearn.load_model(best_model_path)

# Delete runs (DO NOT USE UNLESS CERTAIN)
for run_id in ALL_RUNS_ID:
    client.delete_run(run_id)

# Delete experiment (DO NOT USE UNLESS CERTAIN)
client.delete_experiment(EXPERIMENT_ID)
