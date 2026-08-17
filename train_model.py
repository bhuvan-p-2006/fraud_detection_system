import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("=" * 60)
print("FRAUD DETECTION - MODEL TRAINING")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

DATASET_PATH = "creditcard.csv"

print("\nLoading dataset...")

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print("\nERROR: creditcard.csv was not found.")
    print("Place creditcard.csv in the same folder as train_model.py")
    exit()

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

# ---------------------------------------------------------
# 2. BASIC DATA INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nFraud distribution:")

if "Class" not in df.columns:
    print("\nERROR: Dataset must contain a 'Class' column.")
    exit()

print(df["Class"].value_counts())

# ---------------------------------------------------------
# 3. REMOVE DUPLICATES
# ---------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate records:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# ---------------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nNumber of features:", X.shape[1])

# ---------------------------------------------------------
# 5. TRAIN TEST SPLIT
# ---------------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------------------------
# 6. FEATURE SCALING
# ---------------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 7. TRAIN RANDOM FOREST
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

print("Model training completed!")

# ---------------------------------------------------------
# 8. MAKE PREDICTIONS
# ---------------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------
# 9. EVALUATE MODEL
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# ---------------------------------------------------------
# 10. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Genuine", "Fraud"],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 11. CONFUSION MATRIX
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

print(cm)

# ---------------------------------------------------------
# 12. SAVE MODEL + SCALER + FEATURE NAMES
# ---------------------------------------------------------

print("\nSaving model...")

model_data = {
    "model": model,
    "scaler": scaler,
    "features": list(X.columns)
}

joblib.dump(
    model_data,
    "fraud_model.pkl"
)

print("\nModel saved successfully as:")
print("fraud_model.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)