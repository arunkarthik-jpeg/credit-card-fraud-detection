 Credit Card Fraud Detection App

A machine learning–based web application to detect fraudulent credit card transactions.

 Features
- Detects fraudulent transactions using a trained Random Forest model
- Handles highly imbalanced data using SMOTE
- Web interface built with Streamlit
- Supports large CSV uploads (real transaction data)
- Displays fraud probability and predictions

 Model Details
- Algorithm: Random Forest Classifier
- Imbalance Handling: SMOTE
- Dataset: Credit Card Fraud Dataset (Kaggle)
- Evaluation Metrics: Precision, Recall, F1-score, ROC-AUC

 Live 
 https://creditcardfrauddetection-d9jozrtrnyhzncil2fekhb.streamlit.app/

 Project Structure
 
├── app.py

├── fraud_model_rf.pkl

├── requirements.txt

├── runtime.txt

└── README.md
