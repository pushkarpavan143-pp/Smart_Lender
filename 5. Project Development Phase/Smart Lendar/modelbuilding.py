import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def train_and_save_pipeline():
    # 1. Load Dataset
    data_path = os.path.join('Dataset', 'loan_approval.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # 2. Basic Preprocessing (Handle missing values)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # 3. Encode Categorical Strings to Numbers
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
            
    # ✅ Target column is 'loan_approved'
    if 'loan_approved' not in df.columns:
        raise KeyError("Target column 'loan_approved' not found in dataset.")
        
    X = df.drop(columns=['loan_approved'])
    y = df['loan_approved']
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Define Models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    best_accuracy = 0
    best_model = None
    
    print("="*50)
    print(" MODEL PERFORMANCE EVALUATION ")
    print("="*50)
    
    # 6. Evaluate Loop
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        print(f"{name}:")
        print(f"  Training Accuracy: {train_acc * 100:.1f}%")
        print(f"  Testing Accuracy:  {test_acc * 100:.1f}%\n")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            
    # 7. Serialization using Pickle (no winner printout)
    with open('loan_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("Saved 'loan_model.pkl' successfully to the root folder.")

if __name__ == "__main__":
    train_and_save_pipeline()
