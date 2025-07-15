# Loan Prediction ML Project

This project uses machine learning models to predict whether a loan will be approved or not based on applicant data.

## Objective

To build a classification model that can predict loan approval status (`Y` for approved, `N` for not approved) using applicant information such as income, loan amount, credit history, etc.

## Tools & Libraries Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- XGBoost

## Dataset

The dataset contains information on applicants including:

- Gender, Married, Dependents
- Education, Self_Employed
- ApplicantIncome, CoapplicantIncome
- LoanAmount, Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (target variable)

## Data Preprocessing

- Handled missing values using mean/mode imputation
- Converted categorical variables using label encoding
- Scaled numerical features using StandardScaler

## Exploratory Data Analysis (EDA)

- Used `seaborn` and `matplotlib` for visualization
- Analyzed class distribution, outliers, and feature relationships
- Plotted heatmaps for correlation and confusion matrices

## Models Used

Trained and evaluated the following models:

| Model                | Accuracy | Precision (1) | Recall (1) | F1-Score (1) |
|---------------------|----------|----------------|------------|--------------|
| Logistic Regression | 0.7917   | 0.79           | 0.95       | 0.87         |
| Random Forest       | 0.7750   | 0.81           | 0.89       | 0.85         |
| XGBoost             | 0.7167   | 0.71           | 1.00       | 0.83         |
| KNN                 | 0.7000   | 0.76           | 0.85       | 0.80         |

Best Model Chosen: Logistic Regression — due to highest F1-score and balance between precision and recall.

## Model Evaluation

- Used `accuracy_score`, `confusion_matrix`, and `classification_report`
- Visualized confusion matrix using `seaborn.heatmap`

## Final Prediction

Predicted a sample applicant's loan status using the trained Logistic Regression model.

```python
sample = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(sample)
print("Predicted Loan Status:", "Approved" if prediction[0] == 1 else "Not Approved")


## Conclusion

Logistic Regression was the best-performing model.

Credit history and loan amount were key drivers of prediction.

Built a reliable ML pipeline from preprocessing to deployment-ready predictions.

## Author

Shahnawaz Ansari
Student at Humber Polytechnic
Aspiring Data Analyst

