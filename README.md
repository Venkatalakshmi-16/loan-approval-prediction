Loan Approval Prediction System
Overview

This project predicts whether a loan application will be approved or not using Machine Learning techniques. The model is trained on structured financial and demographic data and deployed using Streamlit for interactive prediction.

Problem Statement

Financial institutions need a reliable system to evaluate loan applications based on applicant details. This project builds a classification model to assist in loan approval decision-making.

Tech Stack

Python

Pandas

NumPy

Scikit-learn

Streamlit

Project Workflow

Data preprocessing

Handling missing values

Encoding categorical variables

Feature scaling

Model training

Logistic Regression

Random Forest

Model evaluation

Accuracy

Precision

Recall

Confusion Matrix

Deployment

Interactive web interface using Streamlit

Project Structure

loan_streamlit_app
│
├── app.py → Streamlit application
├── train.py → Model training script
├── model.pkl → Trained model file
├── scaler.pkl → Feature scaler
├── loan_data.csv → Dataset
├── README.md

How to Run

Clone the repository
git clone <your-repo-link>

Install dependencies
pip install -r requirements.txt

Run the application
streamlit run app.py

Future Improvements