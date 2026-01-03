# 📊 Customer Churn Prediction App

This project predicts whether a customer is likely to churn using machine learning.
It follows a complete data science workflow including EDA, preprocessing, model training,
model persistence using joblib, and deployment as a Streamlit web application.

---

## 🚀 Live Demo
🔗 https://customer-churn-predictor101.streamlit.app/

---

## 📌 Problem Statement
Customer churn refers to customers leaving a service. Predicting churn helps businesses
identify high-risk customers early and take proactive retention measures.

---

## 🧠 Approach

1. **Exploratory Data Analysis (EDA)**  
   - Analyzed customer behavior patterns related to churn  
   - Studied tenure, monthly charges, and contract types  

2. **Data Preprocessing**  
   - Converted customer status into a binary churn label  
   - Removed data leakage columns such as churn reason  
   - Selected relevant numerical and categorical features  

3. **Model Training**  
   - Trained a Logistic Regression classification model  
   - Used separate encoders per categorical feature  
   - Ensured reproducibility using `random_state`  

4. **Model Persistence**  
   - Saved trained model and encoders using `joblib`  
   - Enabled fast inference without retraining  

5. **Streamlit Application**  
   - Built an interactive UI for churn prediction  
   - Displays churn probability (%) and risk level  

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  
- Joblib  

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

📈 Model Output

Churn prediction (Low / Medium / High risk)
Churn probability percentage

🌐 Deployment

The application is deployed using Streamlit Community Cloud and connected
directly to this GitHub repository.
