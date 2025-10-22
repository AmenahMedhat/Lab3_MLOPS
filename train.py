# train.py
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

# ---- Load data ----
train_df = pd.read_csv("data/train.csv")

# Split into features and target
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Initialize MLflow ----
mlflow.set_experiment("Titanic_Classification")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
with mlflow.start_run(run_name="Base_LogisticRegression"):
    # Define and train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Log parameters and metrics
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})

    # Save model locally
    os.makedirs("models", exist_ok=True)
    model_path = "models/logistic_regression_model.pkl"
    joblib.dump(model, model_path)

    # Log as a regular artifact (works on DagsHub)
    mlflow.log_artifact(model_path)

    print(f"✅ Logistic Regression model trained and saved to {model_path}")
    print(f"📊 Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
