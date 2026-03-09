import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Create data directory for processed files
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load data
df = pd.read_csv('data/obesity_data.csv')

# Separate features and target
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, 'models/target_encoder.pkl')

# Define features types
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features).tolist()
all_features = numeric_features + cat_features_encoded

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Save processed data
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)

# Save preprocessor
joblib.dump(preprocessor, 'models/preprocessor.pkl')

print("Preprocessing complete. Processed data and preprocessor saved.")
print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
