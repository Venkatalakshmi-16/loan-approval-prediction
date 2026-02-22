**Loan Approval Prediction Using Machine Learning**
**Overview**

This project predicts whether a loan application will be approved or rejected using Machine Learning. The system helps banks make faster and more consistent loan decisions based on applicant financial details.

The project was developed using Python, Scikit-learn, and Streamlit. Logistic Regression was selected as the final model after comparing multiple algorithms.

**Problem Statement**

Banks receive many loan applications daily. Manual evaluation takes time and may lead to inconsistent or risky decisions. This project builds a classification model to automatically predict loan approval based on applicant details.

**Dataset**

The dataset includes:

Applicant Income

Coapplicant Income

Loan Amount

Credit History

Property Area

Loan Status (Target Variable)

**Loan_Status:**

1 → Approved

0 → Rejected

**Approach**

Data preprocessing steps included handling missing values, encoding categorical variables, creating a new feature called TotalIncome, and applying StandardScaler for feature normalization.

Two models were trained:

Logistic Regression

Random Forest

After evaluation using accuracy and confusion matrix, Logistic Regression performed better and was selected as the final model.

**Deployment**

The trained model and scaler were saved as model.pkl and scaler.pkl.

A Streamlit web application was built where users can enter applicant details and get real-time loan approval predictions with probability.

**How to Run**

Clone the repository

Install dependencies using pip install -r requirements.txt

Run the app using streamlit run app.py

**Conclusion**

This project demonstrates an end-to-end machine learning pipeline, from data preprocessing and model training to deployment using Streamlit for real-time loan approval prediction.