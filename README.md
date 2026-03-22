# 🛡️ Credit Card Fraud Detection System
> **Identifying fraudulent patterns in financial transactions using Advanced Machine Learning.**

This project addresses the "Class Imbalance" problem in financial datasets to detect fraudulent transactions with high accuracy. The system acts as a digital shield, analyzing transaction features to flag suspicious behavior in real-time.

---

## 🧠 Computational Logic: How the "Digital Shield" Works

The algorithm processes transaction data through a specialized security pipeline:

### 1. Handling Extreme Class Imbalance
In fraud detection, 99% of transactions are "Normal." A standard computer brain would simply ignore the 1% of "Fraud" cases. 
*   **Strategy:** I implemented **SMOTE (Synthetic Minority Over-sampling Technique)** or **Under-sampling** to balance the dataset.
*   **Computer Logic:** The algorithm is forced to learn the "rare" patterns of fraud by augmenting the minority class data points.

### 2. PCA Transformation (Anonymized Features)
The dataset uses **Principal Component Analysis (PCA)** features (V1, V2, ... V28) to protect user privacy.
*   **Logic:** The system analyzes the variance and correlation between these transformed features to find "anomalous" clusters that deviate from standard spending habits.

### 3. Evaluation Metrics (Beyond Accuracy)
In fraud detection, **Accuracy is a trap.** If the model says "everything is normal," it gets 99% accuracy but fails its mission.
*   **Metric Focus:** I prioritized **Precision-Recall AUC** and the **F1-Score**.
*   **Goal:** Minimizing *False Negatives* (missing a fraud) while keeping *False Positives* (blocking a real customer) at a manageable level.

---

## 🛠️ Tech Stack
*   **Environment:** Python (Jupyter Notebook)
*   **Libraries:** `Scikit-Learn`, `Pandas`, `Imbalanced-Learn` (for SMOTE), `Seaborn`.
*   **Algorithms:** Random Forest / Logistic Regression / XGBoost [Hangisini kullandıysan seç].
