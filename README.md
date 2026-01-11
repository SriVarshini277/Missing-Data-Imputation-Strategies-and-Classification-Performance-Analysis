# Breast Cancer Diagnosis: Missing Data Imputation and Classification Analysis

This project explores multiple missing data imputation strategies on the **Breast Cancer Wisconsin (Original)** dataset and evaluates their impact on classification performance using **KNN** and **SVM** models.
The goal is to understand how different imputation techniques affect model accuracy, data variance, and information retention when handling incomplete medical datasets.
---

## Dataset
- **Source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
- **Instances:** 699
- **Features:** 9 clinical attributes
- **Target:** Benign vs Malignant
- **Missing Values:** Present in the *Bare Nuclei* feature

---

## Imputation Methods Implemented
1. **Mean Imputation**
   - Missing values replaced with the mean of observed values.
   - Assumes data is Missing Completely At Random (MCAR).

2. **Regression Imputation**
   - Linear regression model trained on complete cases.
   - Missing values predicted using remaining clinical features.

3. **Regression with Perturbation**
   - Adds random noise to regression predictions.
   - Preserves variance and avoids overconfident imputations.

4. **Listwise Deletion**
   - Removes rows containing missing values.
   - Results in reduced dataset size and potential information loss.

5. **Missing Indicator Method**
   - Adds a binary indicator for missingness.
   - Mean imputation applied afterward.

---

## Machine Learning Models
The following classifiers were trained on each imputed dataset:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

---

## Results

| Method                  | KNN Accuracy | SVM Accuracy | Dataset Size |
|-------------------------|-------------|--------------|--------------|
| Mean Imputation         | 0.986       | 0.964        | 699          |
| Regression Imputation   | 0.979       | 0.971        | 699          |
| Regression + Noise      | 0.979       | 0.964        | 699          |
| Listwise Deletion       | 0.949       | 0.949        | 683          |
| Missing Indicator       | 0.986       | 0.964        | 699          |

---

## Key Insights

- Regression-based imputation methods preserved predictive performance while maintaining dataset size.
- Listwise deletion resulted in noticeable accuracy loss due to reduced data.
- Adding a missingness indicator helped models learn patterns related to missing data.
- Differences between imputation methods were small due to low missingness, but would increase with higher missing rates.

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (for analysis & visualization)

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Srivarshini277/Missing-Data-Imputation-Strategies-and-Classification-Performance-Analysis.git
2. Navigate to the project folder:
   ```bash
     cd Missing-Data-Imputation-Strategies-and-Classification-Performance-Analysis 
4. Run the analysis script or notebook:
   ```bash
   python imputation_analysis.py

 ## Conclusion
This project demonstrates that thoughtful handling of missing data can significantly impact classification performance. Regression-based imputation methods provide a strong balance between accuracy and data retention, making them suitable for real-world medical datasets.
