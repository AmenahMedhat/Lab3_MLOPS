import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the full Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# ---- Basic Data Cleaning ----
# Fill missing numeric columns with median
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Fill missing categorical column with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop columns that are not useful for modeling
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

# ---- Encode categorical variables ----
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# ---- Split into train/test ----
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Survived"])

# ---- Save train/test files ----
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Preprocessing complete — train.csv and test.csv saved in data/")
