import pandas as pd
from sklearn.svm import SVC


df = pd.read_csv(r"C:\Users\Dell\Desktop\Python\projects\heart.csv")

# data cleaning
# print(df.head())
# print(df.isnull().sum())

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

X = df[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# prediction
y_pred = model.predict(X_test_scaled)

# evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# joblib save model
import joblib
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(X.columns.tolist(), "heart_columns.pkl")
