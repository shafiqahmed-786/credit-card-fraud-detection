# 💳 Credit Card Fraud Detection using Machine Learning

Credit card fraud causes significant financial losses for banks and customers worldwide.  
This project builds a **machine learning–based fraud detection system** that classifies transactions as **fraudulent** or **legitimate** using historical transaction data.

The focus is on handling **highly imbalanced data**, maximizing fraud detection recall, and evaluating models using business-relevant metrics.

---

## 🚀 Project Objectives

- Detect fraudulent credit card transactions accurately  
- Handle extreme class imbalance effectively  
- Compare multiple ML models  
- Evaluate models using precision, recall, ROC-AUC, and PR-AUC  
- Save the best performing model for future deployment  

---

## 📂 Dataset

- **Source:** Kaggle – Credit Card Fraud Detection  
- **Link:** https://www.kaggle.com/mlg-ulb/creditcardfraud  
- **Description:**
  - Transactions made by European cardholders
  - Features `V1`–`V28` are PCA-transformed
  - `Amount` and `Time` are raw features
  - `Class` → 0: Legitimate, 1: Fraud

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- XGBoost  
- Matplotlib, Seaborn  
- SHAP (model explainability)  
- Google Colab (development environment)

---

## 🧠 Methodology

### 1. Data Loading & Exploration
- Checked missing values and class distribution
- Visualized transaction amounts and imbalance

### 2. Preprocessing
- Standardized `Amount` and `Time`
- Train-test split with stratification
- Applied **SMOTE** only on training data

### 3. Model Training
Trained and compared:
- Logistic Regression  
- Random Forest  
- XGBoost  

### 4. Evaluation Metrics
- Precision, Recall, F1-Score  
- ROC-AUC  
- Precision-Recall AUC (preferred for imbalance)  
- Confusion Matrix  

### 5. Threshold Tuning
- Adjusted probability threshold to improve fraud recall
- Reduced false negatives (missed frauds)

### 6. Explainability
- Feature importance (tree-based models)
- SHAP values for local and global explanations

---

## 📊 Results

- **XGBoost** achieved the best overall performance
- High **Recall** for fraud detection
- Strong **PR-AUC**, suitable for imbalanced classification
- Model effectively identifies high-risk transactions

---

## 💾 Saved Artifacts

- `best_fraud_model.pkl` → trained XGBoost model  
- `scaler_amount_time.pkl` → scaler for Amount & Time  

These files can be used later for deployment or API development.

---

## ▶️ How to Run

1. Open the notebook in **Google Colab**
2. Upload the dataset or download via KaggleHub
3. Run cells sequentially
4. Models train and evaluate automatically
5. Best model is saved locally

---

## 🔮 Future Improvements

- Hyperparameter tuning with cross-validation  
- Time-based validation to prevent data leakage  
- Real-time deployment using FastAPI  
- Fraud monitoring & data drift detection  
- Integration with transaction systems  

---

## 👤 Author

**Shafiq Ahmed**  
- B.Tech CSE | Data Science & GenAI Enthusiast  
- GitHub: https://github.com/shafiqahmed-786  
- LinkedIn: https://linkedin.com/in/shafiq-ahmed-khan  

---

⭐ If you find this project useful, feel free to star the repository!
