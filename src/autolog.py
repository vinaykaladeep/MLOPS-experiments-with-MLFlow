import os
import shutil
import mlflow
import mlflow.sklearn
from mlflow.data import from_numpy
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# --------------------------------------------------
# Initialize DagsHub + MLflow
# --------------------------------------------------
dagshub.init(
    repo_owner='vinaykaladeep',
    repo_name='MLOPS-experiments-with-MLFlow',
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/vinaykaladeep/MLOPS-experiments-with-MLFlow.mlflow"
)

# --------------------------------------------------
# Enable MLflow Auto Logging
# --------------------------------------------------
mlflow.autolog(log_input_examples=False)
mlflow.sklearn.autolog(log_models=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------
max_depth = 18
n_estimators = 50

# --------------------------------------------------
# MLflow Experiment
# --------------------------------------------------
mlflow.set_experiment("MLOPS-experiment3")

# Paths
cwd = os.getcwd()
script_path = os.path.abspath(__file__)
model_dir = os.path.join(cwd, "Random-Forest-MLflow")

# --------------------------------------------------
# Overwrite model folder if exists
# --------------------------------------------------
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

# --------------------------------------------------
# Start MLflow Run
# --------------------------------------------------
with mlflow.start_run():

    # --------------------------------------------------
    # Log dataset (Train + Eval)
    # --------------------------------------------------
    train_ds = from_numpy(features=X_train, targets=y_train)
    test_ds = from_numpy(features=X_test, targets=y_test)

    mlflow.log_input(train_ds, context="train")
    mlflow.log_input(test_ds, context="eval")

    # --------------------------------------------------
    # Train the Model
    # --------------------------------------------------
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Predictions & manual accuracy (autolog still logs its own metric)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # --------------------------------------------------
    # Confusion Matrix (manual artifact)
    # --------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    cm_path = os.path.join(cwd, "Confusion-matrix.png")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(script_path)

    # --------------------------------------------------
    # Save MLflow Model Folder (Overwrite)
    # --------------------------------------------------
    mlflow.sklearn.save_model(sk_model=rf, path=model_dir)

    # Log full model folder as artifact
    mlflow.log_artifacts(model_dir, artifact_path="Random-Forest-Model")

    # --------------------------------------------------
    # Tags
    # --------------------------------------------------
    mlflow.set_tags({
        "Author": "VinayKaladeep",
        "Project": "Wine Classification"
    })

    print(f"Confusion matrix logged: {cm_path}")
    print(f"Model folder saved: {model_dir}")