import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def predict():
    try:
        inputs = []
        for feature, entry in zip(feature_names, entry_widgets):
            value = float(entry.get())
            inputs.append(value)
        
        input_df = pd.DataFrame([dict(zip(feature_names, inputs))])
        
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        result = "POSITIF Diabetes" if prediction == 1 else "NEGATIF Diabetes"
        result_label.config(text=f"Hasil: {result}", 
        bg="lightsalmon" if prediction == 1 else "lightgreen",
        font=("Arial", 12, "bold"))
    except ValueError:
        messagebox.showerror("Error", "Semua nilai harus diisi dengan angka")

root = tk.Tk()
root.title("Prediksi Diabetes")
root.geometry("500x620")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="Prediksi Diabetes dengan Machine Learning", 
font=("Arial", 16, "bold"), bg="#f0f0f0", pady=10)
title_label.pack()

input_frame = ttk.LabelFrame(root, text="Masukkan Data")
input_frame.pack(padx=20, pady=10, fill="x")

entry_widgets = []
for i, feature in enumerate(feature_names):
    row_frame = tk.Frame(input_frame, bg="#f0f0f0")
    row_frame.pack(fill="x", padx=10, pady=5)
    
    label = tk.Label(row_frame, text=f"{feature}:", width=20, anchor="w", bg="#f0f0f0")
    label.pack(side=tk.LEFT)
    
    entry = tk.Entry(row_frame)
    entry.pack(side=tk.LEFT, fill="x", expand=True)
    entry_widgets.append(entry)

button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=20)

predict_button = tk.Button(button_frame, text="Prediksi", command=predict, 
bg="#4CAF50", fg="white", font=("Arial", 12), 
padx=20, pady=10)
predict_button.pack()

result_label = tk.Label(root, text="Hasil akan muncul di sini", 
bg="#f0f0f0", font=("Arial", 12), 
width=30, height=2)
result_label.pack(pady=20)

root.mainloop()