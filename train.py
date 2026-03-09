import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import json
import os

# Create results directory
os.makedirs('results', exist_ok=True)

# Load processed data
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

# Load target encoder to get class names
le = joblib.load('models/target_encoder.pkl')
target_names = le.classes_.tolist()

results = {}

# 1. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

results['Random Forest'] = {
    'f1_macro': f1_score(y_test, y_pred_rf, average='macro'),
    'precision_macro': precision_score(y_test, y_pred_rf, average='macro'),
    'recall_macro': recall_score(y_test, y_pred_rf, average='macro'),
    'report': classification_report(y_test, y_pred_rf, target_names=target_names, output_dict=True)
}

joblib.dump(rf, 'models/random_forest_model.pkl')

# 2. XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

results['XGBoost'] = {
    'f1_macro': f1_score(y_test, y_pred_xgb, average='macro'),
    'precision_macro': precision_score(y_test, y_pred_xgb, average='macro'),
    'recall_macro': recall_score(y_test, y_pred_xgb, average='macro'),
    'report': classification_report(y_test, y_pred_xgb, target_names=target_names, output_dict=True)
}

joblib.dump(xgb, 'models/xgboost_model.pkl')

# Save all results to JSON
with open('results/metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nTraining Complete. Results saved to results/metrics.json")

# Console Summary
for model, metrics in results.items():
    print(f"\n--- {model} ---")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
