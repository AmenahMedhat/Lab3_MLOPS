# tune.py
import pandas as pd
import mlflow
import os
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

# ---- Load preprocessed data ----
train_df = pd.read_csv("data/train.csv")

X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Split into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Define hyperparameter grids ----
n_estimators_list = [50, 100, 200]
max_depth_list = [3, 5, 7]

mlflow.set_experiment("Titanic_Classification_Tuning")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ---- Parent MLflow run ----
with mlflow.start_run(run_name="RandomForest_Tuning") as parent_run:
    for n in n_estimators_list:
        for d in max_depth_list:
            with mlflow.start_run(run_name=f"RF_n{n}_d{d}", nested=True):
                # Define and train model
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred)
                rec = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)

                # Log parameters and metrics
                mlflow.log_param("model_type", "RandomForestClassifier")
                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})

                print(f"Trained RF with n_estimators={n}, max_depth={d} → Accuracy={acc:.4f}")

print("✅ Hyperparameter tuning complete! Check MLflow UI for results.")
