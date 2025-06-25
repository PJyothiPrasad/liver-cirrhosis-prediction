import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import pickle
import os
import pandas as pd

df = pd.read_csv("data/liver.csv")
print(df.columns)


# Load dataset
df = pd.read_csv("data/liver.csv")
df['Dataset'] = df['Dataset'].replace(2, 0)

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df.dropna(inplace=True)

X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Normalize
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and normalizer
with open("rf_acc_68.pkl", "wb") as f:
    pickle.dump(model, f)

with open("normalizer.pkl", "wb") as f:
    pickle.dump(normalizer, f)

print("✅ Model and normalizer saved.")
