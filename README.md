# Heart Disease Prediction

This project predicts the **risk of heart disease** using a machine learning model.  
It uses patient health details as input and provides predictions through a **Streamlit web app**.

---

## Features
- Easy-to-use **Streamlit interface**  
- Input form for age, gender, chest pain type, blood pressure, cholesterol, ECG results, sugar level, and maximum heart rate  
- Data preprocessing using **StandardScaler**  
- Trained machine learning model stored with **Joblib**  
- Real-time prediction of **Low risk** or **High risk**  

---

## Technologies Used
- **Python 3.9+**  
- **Pandas** for data handling  
- **Scikit-learn** for model training and preprocessing  
- **Joblib** for saving and loading models  
- **Streamlit** for building the web app  

---

## Files in the Project
- `heart_model.pkl` → Trained machine learning model  
- `heart_scaler.pkl` → StandardScaler object for preprocessing  
- `heart_columns.pkl` → Feature columns used during training  
- `heartApp.py` → Streamlit app code  
- `heart.csv` → Dataset used for training  
- `README.md` → Project documentation  

