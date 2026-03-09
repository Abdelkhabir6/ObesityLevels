import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Load data
df = pd.read_csv('data/obesity_data.csv')

# 1. Statistical Summary
summary = df.describe(include='all')
summary.to_csv('data/statistical_summary.csv')
print("Statistical summary saved to data/statistical_summary.csv")

# 2. Target Variable Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='NObeyesdad', order=df['NObeyesdad'].value_counts().index, palette='viridis')
plt.title('Distribution of Obesity Levels')
plt.tight_layout()
plt.savefig('plots/target_distribution.png')
plt.close()

# 3. Correlation Heatmap (Numerical features only)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# 4. Age vs Obesity Level (Example of relationship)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Age', y='NObeyesdad', palette='magma')
plt.title('Age Distribution by Obesity Level')
plt.tight_layout()
plt.savefig('plots/age_vs_obesity.png')
plt.close()

print("EDA plots saved to plots/ directory.")
