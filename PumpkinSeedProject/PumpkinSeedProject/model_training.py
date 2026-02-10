import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_excel("dataset/Pumpkin_Seeds_Dataset.xlsx")

print("Dataset Loaded Successfully!")
print(df.head())
print(df.info())

# Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# Encode Target
le = LabelEncoder()
df["Class"] = le.fit_transform(df["Class"])

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Model
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(x_train, y_train)

# Prediction
y_pred = random_forest.predict(x_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("\nRandom Forest Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
pickle.dump(random_forest, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("\nModel saved as model.pkl")
print("Scaler saved as scaler.pkl")
print("Label Encoder saved as label_encoder.pkl")
