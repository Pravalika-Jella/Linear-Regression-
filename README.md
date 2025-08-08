# Linear Regression – Simple & Multiple (California Housing Dataset)

This project demonstrates the implementation and understanding of **Simple Linear Regression** and **Multiple Linear Regression** using Python's **scikit-learn** library. The **California Housing dataset** is used as a case study to explore regression concepts, evaluation metrics, and model interpretation. This work is part of my internship tasks.

---

## 📌 Overview
Linear regression is one of the most fundamental algorithms in machine learning, used to model the relationship between a dependent variable (target) and one or more independent variables (features).

- **Simple Linear Regression:** Uses a single feature to predict the target value and fits a straight line (`y = m*x + c`).
- **Multiple Linear Regression:** Uses multiple features to predict the target, fitting a multidimensional plane.

---

## 🛠 Steps Performed
1. **Import & Preprocess Data** – Loaded the California Housing dataset, checked structure, and selected features.
2. **Train-Test Split** – Divided the dataset into training and testing sets.
3. **Model Training** – Trained both simple and multiple linear regression models using `LinearRegression` from scikit-learn.
4. **Evaluation** – Assessed models with:
   - Mean Absolute Error (**MAE**)
   - Mean Squared Error (**MSE**)
   - Coefficient of Determination (**R² Score**)
5. **Visualization** – 
   - Plotted regression line for simple regression.
   - Compared predicted vs. actual values for multiple regression.
6. **Multicollinearity Check** – Calculated **Variance Inflation Factor (VIF)** for all predictors.

---

## 📊 Evaluation Metrics
- **MAE:** Measures the average magnitude of errors without considering direction.
- **MSE:** Penalizes larger errors more than MAE.
- **R² Score:** Shows how well the features explain the target variable's variance.

---

## 🎯 Learning Outcomes
- Understanding regression modeling in Python.
- Interpreting coefficients in simple and multiple regression.
- Using evaluation metrics to judge performance.
- Detecting multicollinearity using VIF.

---

## 💡 Applications
- Predicting housing prices
- Sales forecasting
- Financial trend analysis
- Any continuous value prediction task

---

## 📎 Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
