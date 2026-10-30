# FinGuard: Credit Card Fraud Detection Platform
### Scalable Big Data Machine Learning with Apache PySpark, Decision Tree & Streamlit

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0-orange.svg)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg)
![OS](https://img.shields.io/badge/OS-Windows%2010%2F11-0078D6.svg)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)

---

## 1. Project Introduction

**FinGuard** is an enterprise-grade, end-to-end Big Data machine learning platform designed to identify fraudulent credit card transactions in real time. Operating on Apache PySpark and Streamlit, the system ingests large-scale transactional data, extracts behavioral and geospatial features, handles severe class imbalance, trains an interpretable Decision Tree Classifier, and presents an interactive financial fraud monitoring dashboard.

The solution is specifically engineered to run seamlessly in local and distributed Windows 10/11 environments, utilizing loopback socket configurations, memory-optimized partition caching, and strict data leakage prevention.

---

## 2. Problem Statement

Modern financial institutions process millions of electronic transactions daily. However, credit card fraud exhibits extreme class imbalance: fraudulent transactions account for less than **0.5%** of all transactions (in our 5,000-record dataset, fraud comprises only **18** transactions or **0.36%**).

Standard classification models often suffer from:
1. **The Accuracy Paradox**: A naive model classifying 100% of transactions as legitimate achieves 99.64% accuracy while catching zero fraudulent events ($0 loss mitigated).
2. **Data Leakage**: Preprocessing parameters (such as IQR outlier caps and category distributions) fit across the entire dataset rather than training folds.
3. **Severe False-Positive Friction**: Overly aggressive blocking mechanisms decline legitimate cardholder purchases, causing frustration and card abandonment.
4. **Latency & Scalability Bottlenecks**: Inability of single-threaded Python scripts to process big data transaction streams.

---

## 3. Project Objectives

- **Big Data Scalability**: Leverage **Apache PySpark MLlib** for distributed data ingestion, schema validation, and feature vectorization.
- **Binary Classification**: Robustly classify transactions into:
  - `0`: Legitimate Transaction
  - `1`: Fraudulent Transaction
- **Class Imbalance Mitigation**: Compute balanced inverse-frequency class weights and stratified oversampling to ensure sensitive split criteria on minority fraud patterns.
- **Leakage-Free IQR Outlier Handling**: Fit interquartile range thresholds strictly on training partitions; clamp legitimate transaction amounts while strictly preserving true fraud values.
- **Interpretability**: Utilize a Decision Tree Classifier to extract actionable split rules and feature importances for banking regulatory audits.
- **Executive Frontend**: Provide a modern dark fintech Streamlit web dashboard for executive KPIs, exploratory analysis, live inference, and behavioral pattern discovery.

---

## 4. Technology Stack

| Layer | Component | Version / Specification |
| :--- | :--- | :--- |
| **Runtime Environment** | Python & Java | Python 3.11.x, OpenJDK 17 (Temurin 17.0.20) |
| **Big Data Engine** | Apache Spark | PySpark 4.2.0 (local[2] execution mode) |
| **Data Processing** | Pandas & NumPy | Pandas 2.2.x, NumPy 2.2.x, PyArrow |
| **Machine Learning** | PySpark MLlib | DecisionTreeClassifier, StringIndexer, OneHotEncoder, VectorAssembler, Pipeline |
| **Evaluation Metrics** | MLlib & Scikit-learn | BinaryClassificationEvaluator, MulticlassClassificationEvaluator, Scikit-learn metrics |
| **Interactive Frontend** | Streamlit | Streamlit 1.58.x (Custom Dark Fintech CSS) |
| **Visualization** | Plotly & Seaborn | Plotly 7.1.x (Interactive), Matplotlib 3.10.x, Seaborn 0.13.x (Static Reports) |

---

## 5. Dataset Description

The application natively supports `fraudTest_5000.csv` and `credit_card_transactions.csv` located in the `data/` directory.

### Dataset Schema & Attributes

| Attribute Name | PySpark Inferred Type | Description |
| :--- | :--- | :--- |
| `trans_date_trans_time` | Timestamp | Timestamp when the transaction took place |
| `cc_num` | BigInt | Credit card account number (dropped to prevent PII exposure) |
| `merchant` | String | Merchant business identifier |
| `category` | String | Merchant category (e.g. `shopping_net`, `grocery_pos`, `travel`) |
| `amt` | Double | Transaction amount billed in USD ($) |
| `first`, `last`, `street` | String | Cardholder PII identity fields (dropped during cleaning) |
| `gender` | String | Cardholder gender (`M` / `F`) |
| `city`, `state`, `zip` | String / Integer | Cardholder residential location |
| `lat`, `long` | Double | Latitude and longitude coordinates of cardholder residence |
| `city_pop` | Double | Population of the cardholder's municipality |
| `job`, `dob` | String | Cardholder occupation and date of birth |
| `trans_num`, `unix_time`| String / BigInt | Transaction ID and epoch timestamp (dropped) |
| `merch_lat`, `merch_long`| Double | Latitude and longitude coordinates of merchant terminal |
| `is_fraud` (Target) | Integer / Double | Binary label: `0` = Legitimate, `1` = Fraudulent |

---

## 6. Folder Structure

```
credit_card_fraud_detection/
├── data/
│   ├── credit_card_transactions.csv  # 5,000 raw real transaction records
│   └── fraudTest_5000.csv            # Auto-detected primary dataset
├── models/
│   ├── decision_tree_model/          # Serialized PySpark PipelineModel
│   ├── metrics.json                  # Accuracy, Precision, Recall, F1, ROC-AUC
│   ├── roc_data.json                 # Precomputed FPR, TPR coordinates
│   └── feature_importance.csv        # Feature rankings & importance percentages
├── reports/
│   └── figures/                      # High-resolution publication plots
│       ├── amount_distribution.png
│       ├── auc_roc_curve.png
│       ├── confusion_matrix.png
│       ├── correlation_heatmap.png
│       ├── feature_importance.png
│       ├── fraud_distribution_by_amount.png
│       ├── fraud_distribution_by_category.png
│       ├── fraud_distribution_by_gender.png
│       ├── fraud_distribution_by_hour.png
│       ├── fraud_distribution_by_state.png
│       └── fraud_vs_legit.png
├── src/
│   ├── __init__.py
│   ├── spark_session.py              # Singleton SparkSession with loopback IP binding
│   ├── preprocessing.py              # PII filtering, imputation, feature engineering, IQR
│   ├── model_training.py             # Stratified split, class weights, Pipeline, DecisionTree
│   ├── evaluation.py                 # Evaluation metrics, ROC curve, trade-off explanation
│   └── eda.py                        # Static Matplotlib & interactive Plotly visual generators
├── app.py                            # Streamlit 5-page dark fintech web platform
├── main.py                           # End-to-end CLI pipeline orchestrator
├── preprocessing.py                  # Root proxy for preprocessing module
├── model_training.py                 # Root proxy for model training module
├── evaluation.py                     # Root proxy for evaluation module
├── requirements.txt                  # Python 3.11 pinned dependencies
├── setup.bat                         # Automated Windows environment installer
├── run.bat                           # Automated Windows application launcher
└── README.md                         # Complete project documentation
```

---

## 7. Data Preprocessing Methodology

1. **Schema Validation & PySpark Ingestion**: Loads CSV with inferred types. Identifies target column dynamically (`is_fraud`, `Class`, or `fraud`).
2. **Deduplication**: Executes `dropDuplicates()` across transaction records to eliminate accidental batch replay submissions.
3. **Missing Value Imputation**:
   - Numerical columns: Missing values imputed with the training median.
   - Categorical columns: Imputed with `"Unknown"`.
4. **PII and High-Cardinality Filtering**: Removes card numbers, names, street addresses, and unique hashes (`trans_num`, `unix_time`) to eliminate target leakage and comply with financial privacy standards.
5. **IQR Outlier Capping (Fitted on Training Split Only)**:
   - Computes:
     $$\text{IQR} = Q_3 - Q_1$$
     $$\text{Upper Threshold} = Q_3 + (3.0 \times \text{IQR})$$
   - Training threshold: **$289.44**.
   - **Fraud Preservation Rule**: Extreme amounts among legitimate transactions ($label = 0$) are clamped at $289.44, while genuine fraudulent transactions ($label = 1$) preserve their actual values to maintain critical fraud signals.

---

## 8. Feature Engineering

Domain-specific features engineered from raw signals:

1. **Temporal Features**:
   - `trans_hour`: Hour of transaction (0–23).
   - `trans_day`, `trans_month`, `trans_dayofweek`.
   - `is_night`: Binary flag ($1$ if hour $\ge 22$ or hour $\le 5$, else $0$).
2. **Geospatial Proximity (Haversine Formula)**:
   Computes direct great-circle distance in kilometers between customer coordinates $(\text{lat}_1, \text{long}_1)$ and merchant terminal $(\text{lat}_2, \text{long}_2)$:
   $$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
   $$d = 2 R \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$
   where $R = 6,371.0\text{ km}$.
3. **Cardholder Demographics**:
   - `age`: Derived from date of birth (`dob`) and transaction timestamp.
4. **PySpark ML Transformation Pipeline**:
   - `StringIndexer`: Indexes categorical strings (`category`, `gender`) with `handleInvalid="keep"`.
   - `OneHotEncoder`: Expands category indices into sparse binary representation vectors.
   - `VectorAssembler`: Assembles all numerical and encoded vectors into a single feature vector column `features`.

---

## 9. Machine Learning Methodology

### Stratified Data Partitioning
- **80% Training / 20% Testing** stratified split.
- Preserves the exact minority class ratio across both splits.

### Class Imbalance Weighting
Because fraud represents only **0.36%** of volume, standard loss functions favor the majority class. The pipeline computes inverse class weights:
$$W_k = \frac{N_{\text{total}}}{2 \times N_k}$$
- Weight for Legit ($W_0$): **~0.57**
- Weight for Fraud ($W_1$): **~3.85**
- The calculated weight column is supplied directly to PySpark's `DecisionTreeClassifier(weightCol="weight")`.

---

## 10. Decision Tree Explanation

A Decision Tree Classifier partitions the feature space into recursive orthogonal hyperplanes by selecting the feature and threshold that maximize node purity.

- **Impurity Metric**: Gini Impurity:
  $$I_G(t) = 1 - \sum_{i=0}^{1} p(i|t)^2$$
- **Hyperparameter Configuration**:
  - `maxDepth = 5`: Restricts tree depth to prevent memorization and overfitting.
  - `maxBins = 32`: Discretizes continuous variables into 32 histogram bins for fast parallel evaluation.
  - `minInstancesPerNode = 2`: Ensures leaves hold sufficient statistical support.
  - `impurity = "gini"`.

---

## 11. Model Evaluation Metrics

Evaluated on the isolated 20% unseen test partition:

| Metric | Score | Domain Explanation |
| :--- | :--- | :--- |
| **Accuracy** | **99.58%** | Fraction of total correct predictions across both classes |
| **ROC-AUC** | **0.6656** | Discriminative ability of predicted probabilities |
| **Fraud Recall** | **33.33%** | Proportion of actual fraud transactions caught |
| **Fraud Precision** | **33.33%** | Proportion of triggered alerts that were true fraud |
| **Weighted F1** | **0.9958** | Overall harmonic mean accounting for support |

### Confusion Matrix (Test Split)

| | Predicted Legit (0) | Predicted Fraud (1) |
| :--- | :---: | :---: |
| **Actual Legit (0)** | **TN = 956** | **FP = 2** |
| **Actual Fraud (1)** | **FN = 2** | **TP = 1** |

---

## 12. AUC-ROC Explanation & Precision-Recall Trade-off

### Receiver Operating Characteristic (ROC-AUC)
The ROC curve plots **True Positive Rate (Recall)** against **False Positive Rate (1 - Specificity)** across all classification thresholds:
$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
An AUC of **0.50** denotes random guessing, whereas an AUC of **1.0** indicates perfect separation. FinGuard achieves an AUC of **0.6656** on test data.

### The Precision-Recall Equilibrium
- **High Recall Strategy**: Minimizes False Negatives (FN). Essential for preventing monetary theft, chargeback penalties, and liability.
- **High Precision Strategy**: Minimizes False Positives (FP). Essential for preventing cardholder embarrassment and checkout card declines.
- **Production Solution**: Multi-tiered thresholding:
  - Probability $\ge 85\%$: Auto-block card.
  - Probability $40\% - 85\%$: Request SMS OTP or biometric step-up authentication.
  - Probability $< 40\%$: Approve transaction silently.

---

## 13. Dashboard Functionality (Streamlit App)

The Streamlit dashboard (`app.py`) provides real-time multi-factor slice-and-dice intelligence and 5 interactive pages:

### Real-Time Sidebar Filters & Live Telemetry KPIs
- **Live Sidebar KPIs (Real-Time)**:
  - **Active Cohort / Total Records**: Dynamic ingestion share percentage.
  - **Fraud Incidents**: Live fraud count and fraud rate with color-coded risk badge (Green / Amber / Red).
  - **Legitimate Volume**: Live approved transactions count.
  - **Filtered Billed Value**: Total dollars in the active cohort and average transaction size.
  - **Value At Risk**: Intercepted fraudulent dollar value.
- **Dynamic Multi-Factor Sidebar Filters**:
  - **Transaction Class**: Filter by All, Legitimate Only (0), or Fraudulent Only (1).
  - **Merchant Category**: Multi-select across all 14 categories.
  - **Transaction Amount Slider**: Min-max slider ($0 to $2,500+).
  - **Hour of Day Range (0–23h)**: Dual-handle 24-hour time-of-day slider.
  - **Customer-Merchant Distance (km)**: Dual-handle slider (0 to 120 km).
  - **Cardholder Gender**: Filter by Female (F), Male (M), or All.
  - **One-Click Reset**: Resets all filters back to full ingestion defaults.

### Platform Navigation Pages
1. **📊 Overview**: Executive KPI summary cards, live class distribution donut chart, merchant category fraud breakdown, spending distribution histogram, and live audit log stream with **CSV Export**.
2. **🔍 Data Exploration**: Multi-dimensional correlation scatter plot (Amount vs Distance), interactive IQR outlier parameter tester with live cutoff line, dataset preview, schema inspection, and missing value quality audit.
3. **⚡ Fraud Prediction**: Live transaction risk scoring form with 4 instant scenario presets (*Midnight Online Shopping*, *Routine Supermarket*, *Late Night Interstate Gas*, *Local Restaurant*), Plotly Speedometer Risk Gauge (0-100%), decision path driver breakdown, and automated banking policy actions.
4. **📈 Model Performance**: Interactive Decision Threshold Simulator with live operating threshold slider ($\tau \in [0.05, 0.95]$), dynamic Confusion Matrix heatmap, interactive Plotly ROC Curve, Gini feature importance rankings, and precision-recall trade-off documentation.
5. **🔬 Fraud Analytics**: Offline Plotly USA Geospatial Scatter Map, 2D Hour-of-Day vs Day-of-Week Fraud Danger Heatmap, class boxplots, and high-risk behavioral pattern cards.

---

## 14. Installation Instructions

### Prerequisites
- **Operating System**: Windows 10 or Windows 11
- **Python**: Python 3.11.x (64-bit)
- **Java**: OpenJDK 17 (Temurin 17.0.20 or Oracle JDK 17)

### Verification Commands
Open **Windows PowerShell** and verify runtimes:

```powershell
python --version
# Expected: Python 3.11.x

java -version
# Expected: openjdk version "17.x.x"
```

### Automated Setup
Double-click `setup.bat` or run:

```cmd
setup.bat "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"
```

### Manual PowerShell Setup
If preferred, set up manually:

```powershell
# 1. Allow script execution in current PowerShell session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. Create Python 3.11 virtual environment
& "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 15. Execution Instructions

### Option A: Complete Machine Learning Pipeline (CLI)
Trains the model, generates all evaluation metrics, exports artifacts, and creates figures in `reports/figures/`:

```powershell
python main.py
```

### Option B: Launch the Interactive Streamlit Dashboard
Launches the dark fintech web application at `http://localhost:8501`:

```powershell
# Using the Windows batch script:
run.bat

# Or directly from PowerShell:
python -m streamlit run app.py
```

---

## 16. Troubleshooting & Windows Notes

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `PermissionError: [WinError 10013]` | Windows blocked local loopback socket for Py4J | Ensure `spark.driver.bindAddress` is set to `127.0.0.1` (handled automatically in `src/spark_session.py`). |
| `Scripts\Activate.ps1 cannot be loaded` | PowerShell execution policy restricted | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating `.venv`. |
| `JAVA_HOME not set` | Apache Spark cannot find the JVM | Set environment variable: `[System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Eclipse Adoptium\jdk-17.x", "User")`. |
| `isnan() DataType Mismatch` | PySpark `isnan()` called on timestamp/string | Only check `isnan()` on `DoubleType` and `FloatType` columns; use `isNull()` for other data types (implemented in `src/preprocessing.py`). |

---

## 17. License and Acknowledgements

- **Dataset**: Credit Card Fraud Detection demonstration dataset based on the synthetic Kaggle fraud dataset.
- **Engine**: Apache Spark / PySpark MLlib.
- **Author**: Senior Data Scientist & Big Data ML Engineer.
#   c r e d i t - c a r d - f r a u d - d e t e c t i o n  
 