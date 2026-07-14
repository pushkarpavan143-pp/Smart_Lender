import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print(" Starting XGBoost Model Training Engine...")

# Automatically locate the exact directory where this train_model.py script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'Dataset', 'loan_approval.csv')

# Load your new dataset using the bulletproof path rule
try:
    df = pd.read_csv(csv_path)
    print(f" Dataset loaded successfully from: {csv_path}")
except FileNotFoundError:
    print(f" Error: Could not find 'loan_approval.csv' at {csv_path}")
    print(" Action required: Make sure your 'loan_approval.csv' file is physically placed inside that exact folder!")
    exit()

# Data Preprocessing
X = df.drop(columns=['name', 'city', 'loan_approved'])
y = df['loan_approved'].astype(int)

print(f" Features being trained on: {list(X.columns)}")

# Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train the XGBoost Classifier
print(" Training the XGBoost Model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Evaluate the Model Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" Model Accuracy Score: {accuracy * 100:.2f}%")

# Save the trained model to disk as a pickle file in the exact same folder
model_pickle_path = os.path.join(BASE_DIR, 'loan_model.pkl')
with open(model_pickle_path, "wb") as f:
    pickle.dump(model, f)

print(f" Success! 'loan_model.pkl' generated at: {model_pickle_path}")