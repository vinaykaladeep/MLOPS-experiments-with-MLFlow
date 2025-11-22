import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# ----------------------------
# Initialize DagsHub / MLflow
# ----------------------------
dagshub.init(
    repo_owner='vinaykaladeep',
    repo_name='MLOPS-experiments-with-MLFlow',
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/vinaykaladeep/MLOPS-experiments-with-MLFlow.mlflow"
)

# ----------------------------
# Load Wine dataset
# ----------------------------
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# ----------------------------
# Random Forest hyperparameters
# ----------------------------
max_depth = 18
n_estimators = 50

# ----------------------------
# MLflow experiment
# ----------------------------
mlflow.set_experiment('MLOPS-experiment3')

# Working directories
cwd = os.getcwd()
script_path = os.path.abspath(__file__)
model_dir = os.path.join(cwd, "Random-Forest-MLflow")

# ----------------------------
# Delete existing folder if any
# ----------------------------
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

# ----------------------------
# Start MLflow run
# ----------------------------
with mlflow.start_run():

    # Train Random Forest model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ----------------------------
    # Log metrics & parameters
    # ----------------------------
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # ----------------------------
    # Confusion matrix
    # ----------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(cwd, "Confusion-matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Log artifacts
    mlflow.log_artifact(cm_path)       # Confusion matrix
    mlflow.log_artifact(script_path)   # This script

    # ----------------------------
    # Save MLflow model (overwrite)
    # ----------------------------
    mlflow.sklearn.save_model(sk_model=rf, path=model_dir)

    # Log the MLflow model folder as artifacts
    mlflow.log_artifacts(model_dir, artifact_path="Random-Forest-Model")

    # ----------------------------
    # Tags
    # ----------------------------
    mlflow.set_tags({
        "Author": "VinayKaladeep",
        "Project": "Wine Classification"
    })

    # ----------------------------
    # Print summary
    # ----------------------------
    print("Accuracy:", accuracy)
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"MLflow model folder saved/overwritten at: {model_dir}")