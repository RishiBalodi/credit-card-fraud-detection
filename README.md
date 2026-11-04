# 🛡️ FinGuard – Credit Card Fraud Detection Platform

### Big Data Machine Learning using Apache PySpark, Decision Tree & Streamlit

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-orange?logo=apachespark)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/ML-Decision%20Tree-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Overview

**FinGuard** is an end-to-end Credit Card Fraud Detection platform developed using Python, Apache PySpark, Machine Learning, and Streamlit.

The project aims to identify potentially fraudulent credit card transactions by analyzing transaction amounts, merchant categories, transaction timings, geographical distances, and customer behavior.

Using **Apache PySpark MLlib and a Decision Tree Classifier**, the system processes transaction data, performs feature engineering, handles class imbalance, and predicts whether a transaction is legitimate or fraudulent.

An interactive Streamlit dashboard provides fraud prediction, data exploration, model evaluation, and analytical visualizations.

## 🎯 Key Features

- Distributed data processing using Apache PySpark.
- Automated data cleaning and preprocessing.
- Feature engineering using temporal and geographical attributes.
- Handling highly imbalanced transaction datasets.
- Decision Tree classification using PySpark MLlib.
- Model evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Interactive fraud prediction dashboard.
- Fraud analytics and behavioral pattern visualization.
- CSV export and transaction monitoring.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Big Data Processing | Apache Spark 4.2.0, PySpark |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | PySpark MLlib, Scikit-learn |
| Algorithm | Decision Tree Classifier |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web Framework | Streamlit |
| Runtime | OpenJDK 17 |

## 📂 Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   ├── credit_card_transactions.csv
│   └── fraudTest_5000.csv
│
├── models/
│   ├── decision_tree_model/
│   ├── metrics.json
│   ├── roc_data.json
│   └── feature_importance.csv
│
├── reports/
│   └── figures/
│
├── src/
│   ├── spark_session.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── eda.py
│
├── app.py
├── main.py
├── preprocessing.py
├── model_training.py
├── evaluation.py
├── run_pipeline.py
├── requirements.txt
├── packages.txt
├── Dockerfile
├── setup.bat
├── run.bat
└── README.md
