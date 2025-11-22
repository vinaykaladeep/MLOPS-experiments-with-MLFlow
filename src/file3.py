import os
import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub

# Initialize DagsHub / MLflow integration
dagshub.init(repo_owner='vinaykaladeep', repo_name='MLOPS-experiments-with-MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/vinaykaladeep/MLOPS-experiments-with-MLFlow.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 5
n_estimators = 10

# Mention your experiment below
mlflow.set_experiment('MLOPS-experiment3')

# Current working directory
cwd = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))

with mlflow.start_run():
    # Train the model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and params
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(cwd, "Confusion-matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Log artifacts
    mlflow.log_artifact(cm_path)                       # Confusion matrix
    mlflow.log_artifact(os.path.abspath(__file__))     # This script

    # Save and log model
    model_path = os.path.join(cwd, "Random-Forest-Model.pkl")
    joblib.dump(rf, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    # Tags
    mlflow.set_tags({"Author": "VinayKaladeep", "Project": "Wine Classification"})

    print("Accuracy:", accuracy)