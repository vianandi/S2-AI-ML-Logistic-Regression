import numpy as np
import joblib

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

print("=== Uji Prediksi Data Baru ===")
input_data = []

for feature in feature_names:
    val = float(input(f"Masukkan nilai untuk {feature}: "))
    input_data.append(val)

input_np = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_np)

prediction = model.predict(input_scaled)
print(f"\nHasil Prediksi: {'POSITIF Diabetes' if prediction[0] == 1 else 'NEGATIF Diabetes'}")