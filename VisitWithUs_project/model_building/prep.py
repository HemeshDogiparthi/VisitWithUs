
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('VisitWithUs_project/data/tourism.csv')

# Data Cleaning
# Drop CustomerID and Designation columns as they are not needed for modeling
df = df.drop(columns=['CustomerID', 'Designation'])

# Handle missing values (e.g., fill with mode or median)
# For categorical columns, fill with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# For numerical columns, fill with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Ensure target variable is numeric (0 or 1)
df['ProdTaken'] = df['ProdTaken'].astype(int)

# Split data into features (X) and target (y)
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save processed dataframes to files
X_train.to_csv('VisitWithUs_project/data/X_train.csv', index=False)
X_test.to_csv('VisitWithUs_project/data/X_test.csv', index=False)
y_train.to_csv('VisitWithUs_project/data/y_train.csv', index=False)
y_test.to_csv('VisitWithUs_project/data/y_test.csv', index=False)

print("Data loaded, cleaned, and split into training and testing sets. Files saved in VisitWithUs_project/data/")
