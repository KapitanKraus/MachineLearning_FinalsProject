# ChurnShield — Machine Learning Customer Churn Decision Support System
 
**Marc Kraus S. Angeles**
*Introduction to Machine Learning Finals Project*
 
---
 
## 1. Introduction
 
Customer churn is a phenomenon of customers discontinuing the use of a company's product or service, and is one of the most pressing challenges facing businesses in competitive industries. In sectors such as telecommunications, banking, and subscription-based services, retaining an existing customer is significantly more cost-effective than acquiring a new one. Research consistently shows that it costs between 5 to 25 times more to attract a new customer than to keep a current one, and that a mere 5% increase in customer retention can increase profits by 25% to 95%.
 
This software is a machine learning-based decision support system designed for business managers for this telecommunication company. By training predictive models on historical customer data, it becomes possible to flag customers who exhibit behavioral patterns associated with churn, recommend targeted interventions, and quantify the financial consequences of inaction.
 
---
 
## 2. Objectives
 
### 2.1 General Objective
 
To develop a machine learning decision support system for customer churn analysis and retention planning in a business context.
 
### 2.2 Specific Objectives
 
1. To collect, clean, and preprocess a real-world customer churn dataset suitable for machine learning.
2. To train three separate machine learning models, each with a distinct analytical purpose.
3. To predict whether a given customer is likely to churn (binary classification: YES or NO).
4. To recommend an appropriate retention strategy for customers identified as at-risk.
5. To assess the revenue risk level associated with potentially losing a given customer.
6. To evaluate and compare the performance of each model using standard metrics.
 
---
 
## 3. Scope and Limitations
 
### 3.1 Scope
 
The following are included within the boundaries of this project:
 
- Use of a publicly available, real-world customer churn dataset (Telco Customer Churn from Kaggle).
- Training and evaluation of three machine learning models using Python and Scikit-learn.
- A user-facing form that accepts 19 customer attributes as input.
- A results dashboard displaying all three model outputs in a clear, color-coded format.
- Model persistence using Joblib, enabling reuse without retraining.
 
### 3.2 Limitations
 
The following are outside the scope of this project:
 
- Not Connected to the telecommunication company
- Uses publicly available historical data and does not accurately reflect the actual company
- Educational Purposes only
- GUI is only inside the Google Colab notebook
 
---
 
## 4. Dataset
 
### 4.1 Dataset Description
 
The dataset used in this project is the Telco Customer Churn dataset, publicly available on Kaggle (published by user blastchar). It simulates the customer records of a fictional telecommunications company and is one of the most widely used benchmark datasets for churn prediction tasks in the machine learning community.
 
| Property | Details |
|---|---|
| Dataset Name | Telco Customer Churn |
| Source | Kaggle — blastchar/telco-customer-churn |
| Total Records | 7,043 customer entries |
| Total Columns | 21 (20 features + 1 target) |
| Target Variable | Churn (Yes / No) |
| Task Type | Binary Classification |
| Class Distribution | Approximately 73.5% No Churn, 26.5% Yes Churn |
| License | Open / Public Use |
 
### 4.2 Key Features
 
The dataset contains 20 input features grouped into three categories:
 
**Demographic Features**
- `gender` — Customer's gender (Male or Female)
- `SeniorCitizen` — Whether the customer is a senior citizen (0 = No, 1 = Yes)
- `Partner` — Whether the customer has a partner (Yes / No)
- `Dependents` — Whether the customer has dependents (Yes / No)
 
**Service Features**
- `tenure` — Number of months the customer has been with the company
- `PhoneService` — Whether the customer has phone service
- `MultipleLines` — Whether the customer has multiple lines
- `InternetService` — Type of internet service (DSL, Fiber optic, or None)
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` — Add-on service subscriptions
- `StreamingTV`, `StreamingMovies` — Whether the customer streams entertainment content
 
**Billing Features**
- `Contract` — Type of contract (Month-to-month, One year, Two year)
- `PaperlessBilling` — Whether the customer uses paperless billing
- `PaymentMethod` — How the customer pays (Electronic check, Mailed check, Bank transfer, Credit card)
- `MonthlyCharges` — Current monthly charge amount in USD
- `TotalCharges` — Total amount charged to the customer over their tenure
 
---
 
## 5. Machine Learning Techniques
 
ChurnShield uses three distinct machine learning algorithms, each assigned to a different analytical task. This multi-model architecture ensures that the system produces richer, more actionable insights than a single model could provide.
 
### 5.1 Model 1 — Churn Prediction (Logistic Regression)
 
**Purpose**
 
It is the most interpretable algorithm for binary classification. Each feature's weight directly indicates its influence on the outcome.
 
**Output**
 
Model 1 produces two outputs: a binary prediction (YES or NO) and a churn probability percentage. For example: Churn = YES, Probability = 74%.
 
### 5.2 Model 2 — Retention Recommendation (Decision Tree)
 
**Purpose**
 
The labels used to train this model were derived from business logic (e.g., new customers with high charges get "Offer Discount"), and a Decision Tree is perfectly suited to learn and generalize these patterns.
 
**Output**
 
Since no pre-existing "retention action" labels exist in the dataset, labels were engineered using the following business rules applied to the training data:
 
| Label | Action | Rule Condition |
|---|---|---|
| 0 | Offer Discount | tenure <= 12 AND MonthlyCharges >= 70 |
| 1 | Upgrade Plan | Contract = Month-to-month |
| 2 | Priority Support | tenure > 24 AND MonthlyCharges < 50 |
| 3 | No Action Needed | All other customers |
 
### 5.3 Model 3 — Revenue Risk Assessment (Random Forest)
 
**Purpose**
 
- Random Forest is more accurate than a single Decision Tree, making it well-suited for the most financially consequential output in the system.
- It handles both numeric and categorical data effectively after encoding.
 
**Output**
 
Revenue risk levels were derived from each customer's MonthlyCharges, representing the ongoing financial value of that customer to the business:
 
- **HIGH (label 2)** — MonthlyCharges >= $80 — losing this customer results in significant monthly revenue loss.
- **MEDIUM (label 1)** — MonthlyCharges between $50 and $79 — moderate revenue impact.
- **LOW (label 0)** — MonthlyCharges < $50 — lower financial stakes if the customer churns.
 
---
 
## 6. System Architecture
 
| Step | Title | Purpose |
|---|---|---|
| 1 | Install & Import Libraries | Installs all required packages and imports them into the session |
| 2 | Load the Dataset | Fetches the Telco Churn CSV from Google Drive |
| 3 | Exploratory Data Analysis | Prints dataset info and renders three matplotlib charts: churn distribution, monthly charges, and tenure |
| 4 | Preprocessing | Cleans data, applies Label Encoding and StandardScaler, and performs the train/test split |
| 5 | Model 1 — Logistic Regression | Trains the churn prediction model, evaluates it, and plots a confusion matrix |
| 6 | Model 2 — Decision Tree | Generates retention labels from business rules and trains the recommendation model |
| 7 | Model 3 — Random Forest | Generates risk labels from MonthlyCharges and trains the risk assessment model, then plots feature importances |
| 8 | Model Comparison | Displays a bar chart and summary table comparing all three model accuracies |
| 9 | Save Models | Serializes all three models and the scaler to .pkl files using Joblib |
| 10 | Interactive Dashboard | Renders the ipywidgets customer input form and displays HTML prediction results on button click |
| 11 | Random Customer Analyzer | Picks a real customer from the dataset at random, runs all models, and shows actual vs predicted verdict |
 
---
 
## 7. Tools and Technologies
 
| Tool / Library | Purpose in ChurnShield |
|---|---|
| Python | Core programming language for all data processing, model training, and UI logic |
| Google Colab | Free cloud-based Jupyter environment — provides GPU/CPU compute, file storage, and widget rendering with no local setup required |
| Scikit-learn | Machine learning library — provides LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, LabelEncoder, StandardScaler, train_test_split, accuracy_score, classification_report, and confusion_matrix |
| Pandas | Data loading from URL, cleaning, type conversion, and DataFrame manipulation throughout the preprocessing pipeline |
| NumPy | Numerical array construction used when building the input vector for model prediction in the dashboard |
| Joblib | Serializes trained models and the fitted scaler to .pkl files for reuse without retraining |
| Matplotlib | Renders the EDA charts (churn distribution, monthly charges histogram, tenure histogram), the model accuracy bar chart, and the feature importance chart |
| Seaborn | Renders the confusion matrix heatmaps for Model 1 using a color-coded annotation grid |
| ipywidgets | Provides the interactive UI components inside the notebook — dropdowns, number inputs, buttons, and output areas for the customer dashboard |
| IPython.display | Renders styled HTML strings as formatted output inside Colab cells, used to display the prediction results dashboard |
 
---
 
## 8. Output
 
### Model Accuracy Comparison
 
| Model | Algorithm | Accuracy |
|---|---|---|
| Model 1 — Churn Prediction | Logistic Regression | 79.9% |
| Model 2 — Retention Recommendation | Decision Tree | 99.9% |
| Model 3 — Revenue Risk Assessment | Random Forest | 100.0% |
 
### Model 1 — Confusion Matrix (Logistic Regression)
 
|  | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 921 | 114 |
| **Actual: Churn** | 169 | 205 |
 
### Model 2 — Retention Recommendation (Decision Tree) — Accuracy: 99.86%
 
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Offer a Discount | 0.99 | 1.00 | 0.99 | 177 |
| Upgrade Their Plan | 1.00 | 1.00 | 1.00 | 602 |
| Provide Priority Support | 1.00 | 1.00 | 1.00 | 179 |
| No Action Needed | 1.00 | 1.00 | 1.00 | 451 |
| **Macro Avg** | 1.00 | 1.00 | 1.00 | 1409 |
| **Weighted Avg** | 1.00 | 1.00 | 1.00 | 1409 |
 
### Top 10 Feature Importances — Model 3 (Random Forest)
 
| Rank | Feature | Importance Score |
|---|---|---|
| 1 | MonthlyCharges | ~0.50 |
| 2 | InternetService | ~0.13 |
| 3 | StreamingTV | ~0.08 |
| 4 | StreamingMovies | ~0.07 |
| 5 | TotalCharges | ~0.06 |
| 6 | PhoneService | ~0.04 |
| 7 | MultipleLines | ~0.03 |
| 8 | OnlineBackup | ~0.03 |
| 9 | DeviceProtection | ~0.02 |
| 10 | OnlineSecurity | ~0.02 |
