# 💳 Credit Card Fraud Detection using Machine Learning

This project focuses on building a machine learning model to *detect fraudulent credit card transactions. As part of my **2-month internship at Intern Orbit*, this task gave me hands-on experience with real-world financial datasets, class imbalance challenges, and model evaluation techniques.

---

## 🔍 Project Overview

*Prompt*:  
> Develop a machine learning model to accurately detect fraudulent credit card transactions.

*Goal*:
- Preprocess and normalize transaction data
- Handle class imbalance using oversampling techniques (SMOTE)
- Train and evaluate models (Logistic Regression & Random Forest)
- Assess performance using classification metrics (Precision, Recall, F1-score, ROC-AUC)

---

## 📁 Dataset

- *Source*: [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- *Size*: 284,807 transactions with 492 frauds (highly imbalanced)
- *Features*:
  - V1 to V28 (PCA components)
  - Amount, Time (manually scaled)
  - Class (Target: 0 = Normal, 1 = Fraud)

---

## ⚙️ Tech Stack

- *Language*: Python  
- *Libraries*:
  - pandas, numpy, matplotlib, seaborn
  - scikit-learn (Logistic Regression, Random Forest, metrics)
  - imblearn (SMOTE for oversampling)

---

## 🧠 ML Workflow

### 1. Data Preprocessing
- Normalized Amount and Time using StandardScaler
- Dropped unnormalized Amount and Time columns

### 2. Handling Class Imbalance
- Applied *SMOTE (Synthetic Minority Oversampling Technique)* to balance the dataset

### 3. Train-Test Split
- 70% training and 30% testing using train_test_split
- Stratified split to maintain class balance

### 4. Model Training
- *Logistic Regression* – A linear classifier, used as a baseline model
- *Random Forest* – An ensemble model for improved accuracy

### 5. Evaluation Metrics
- *Confusion Matrix*
- *Classification Report* (Precision, Recall, F1-score)
- *ROC-AUC Score*

---

## 📊 Results

| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~97%    | 0.97      | 0.97   | 0.97     | 0.989   |
| Random Forest       | ~98–99% | 0.98      | 0.98   | 0.98     | 0.999   |

✅ *Random Forest* performed the best, achieving near-perfect ROC-AUC and overall metrics.

---

## 📌 Project Benefits

This project allowed me to:
- Address *real-world problems* like fraud detection
- Explore *imbalanced classification challenges*
- Gain practical experience with *model evaluation metrics*
- Learn how *oversampling techniques* improve model performance

---

## 🎥 Demo Video

📺 Watch the complete project walkthrough here:  
🔗 [LinkedIn Video Link](https://www.linkedin.com/in/yourprofile) (Replace with your video link)

---

## 📦 Installation & Usage

```bash
# Clone the repo
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the script
python fraud_detection.py
