LIST PROMPT TERMINAL

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows) #
venv\Scripts\activate

# Install core packages #
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# For Streamlit GUI (optional) #
pip install streamlit

# Install all packages from requirements file (if you have one) #
pip install -r requirements.txt

# Run the main console application #
python main.py

# Run Tkinter GUI version #
python gui_tkinter.py

# Run Streamlit GUI version (if installed) #
streamlit run gui_streamlit.py

# If you see errors about missing modules: #
pip install <module_name>

# If you have issues with joblib: #
pip install joblib

# If you encounter issues with scikit-learn: #
pip install -U scikit-learn

# Deactivate virtual environment when finished #
deactivate

# Make sure diabetes.csv is in your project folder
# You can download from Kaggle if needed:
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

# Export trained model to share (already handled in code)
# The code already saves model to diabetes_model.pkl

# To use the model elsewhere, copy both:
# - diabetes_model.pkl
# - scaler.pkl
