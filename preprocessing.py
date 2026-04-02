import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("customer_data.csv")

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Convert categorical to numeric
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Purchased'] = le.fit_transform(df['Purchased'])

# Feature & target split
X = df[['Age', 'Gender', 'Income']]
y = df['Purchased']

print("Features:\n", X.head())
print("Target:\n", y.head())

# Save processed data
df.to_csv("processed_data.csv", index=False)