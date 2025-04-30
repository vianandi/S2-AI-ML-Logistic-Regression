import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

def train_model():
    print("=== Training Diabetes Prediction Model ===")
    df = pd.read_csv("diabetes.csv")
    print("Data Sample:")
    print(df.head())

    plt.figure(figsize=(10, 6))
    df.hist(bins=20, figsize=(10, 8), color='skyblue', edgecolor='black')
    plt.suptitle("Distribusi Fitur Diabetes Dataset")
    plt.tight_layout()
    plt.show()

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    joblib.dump(model, "diabetes_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model dan scaler berhasil disimpan.")
    
def predict():
    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    
    if not os.path.exists("diabetes_model.pkl") or not os.path.exists("scaler.pkl"):
        print("Model belum dilatih. Silakan latih model terlebih dahulu.")
        return
        
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    print("=== Uji Prediksi Data Baru ===")
    input_data = []

    for feature in feature_names:
        val = float(input(f"Masukkan nilai untuk {feature}: "))
        input_data.append(val)

    input_np = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_np)

    prediction = model.predict(input_scaled)
    print(f"\nHasil Prediksi: {'POSITIF Diabetes' if prediction[0] == 1 else 'NEGATIF Diabetes'}")

if __name__ == "__main__":
    while True:
        print("\n=== Diabetes Prediction System ===")
        print("1. Train Model")
        print("2. Make Prediction")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            train_model()
        elif choice == '2':
            predict()
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")