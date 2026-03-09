import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Load data and encoder
y_test = np.load('data/processed/y_test.npy')
le = joblib.load('models/target_encoder.pkl')
target_names = le.classes_

# 1. Confusion Matrices
models_to_test = {
    'Random Forest': 'models/random_forest_model.pkl',
    'XGBoost': 'models/xgboost_model.pkl'
}

X_test = np.load('data/processed/X_test.npy')

for name, path in models_to_test.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    image_path = f"plots/cm_{name.lower().replace(' ', '_')}.png"
    plt.savefig(image_path)
    plt.close()
    print(f"Confusion matrix for {name} saved to {image_path}")

# 2. Feature Importance (using XGBoost as example)
xgb_model = joblib.load('models/xgboost_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Get feature names
numeric_features = preprocessor.transformers_[0][2]
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
categorical_features = preprocessor.transformers_[1][2]
cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_features = numeric_features + cat_features_encoded

importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices][:15], y=np.array(all_features)[indices][:15], palette='viridis')
plt.title('Top 15 Feature Importances (XGBoost)')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()
print("Feature importance plot saved to plots/feature_importance.png")
