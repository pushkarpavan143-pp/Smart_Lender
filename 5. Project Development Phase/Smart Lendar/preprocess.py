import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Define base directories and load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
csv_path = os.path.join(BASE_DIR, 'dataset', 'loan_approval.csv')

print(f" Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# Drop identifier/text columns that aren't numeric features for ML training
X = df.drop(columns=['name', 'city', 'loan_approved'], errors='ignore')
y = df['loan_approved']

print(f"Initial dataset shape: {X.shape}")

# =====================================================================
# 1. CHECKING AND HANDLING MISSING VALUES
# =====================================================================
print("\n--- Step 1: Checking & Handling Missing Values ---")
missing_summary = X.isnull().sum()
print("Missing values per column before handling:")
print(missing_summary)

# Handling missing values: Fill numeric columns with their median value
for col in X.columns:
    if X[col].isnull().sum() > 0:
        median_value = X[col].median()
        X[col].fillna(median_value, inplace=True)
        print(f"-> Filled missing values in '{col}' with median: {median_value}")

# =====================================================================
# 2. SPLITTING DATA INTO TRAINING AND TESTING SETS
# =====================================================================
# CRITICAL: Always split your data BEFORE scaling or oversampling (SMOTE)
# to completely prevent data leakage from the test set into your training.
print("\n--- Step 2: Splitting Data into Train and Test Sets ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training Features Shape: {X_train.shape} | Testing Features Shape: {X_test.shape}")

# =====================================================================
# 3. BALANCING THE DATASET (SMOTE)
# =====================================================================
print("\n--- Step 3: Balancing the Class Distribution (SMOTE) ---")
print(f"Class distribution before SMOTE:\n{y_train.value_counts()}")

# Initialize SMOTE to balance the training subset exclusively
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Class distribution after SMOTE:\n{y_train_balanced.value_counts()}")

# =====================================================================
# 4. SCALING THE DATA
# =====================================================================
print("\n--- Step 4: Scaling the Feature Sets ---")
scaler = StandardScaler()

# Fit on the balanced training data, then transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print(" Preprocessing Complete! Your data arrays are optimized and ready for training.")
print(f"Final Processed Training Array Shape: {X_train_scaled.shape}")
print(f"Final Processed Testing Array Shape: {X_test_scaled.shape}")