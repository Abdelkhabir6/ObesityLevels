from ucimlrepo import fetch_ucirepo 
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Fetch dataset 
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
  
# Data (as pandas dataframes) 
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets 

# Combine features and targets for full dataset
df = pd.concat([X, y], axis=1)

# Save to CSV
output_path = 'data/obesity_data.csv'
df.to_csv(output_path, index=False)

print(f"Dataset saved to {output_path}")
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
