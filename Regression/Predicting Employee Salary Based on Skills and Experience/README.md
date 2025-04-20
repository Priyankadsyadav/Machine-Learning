# Salary Prediction Model

## Overview

This project is a machine learning model built to predict the salary of individuals based on various factors like experience, education level, location, certifications, and skill match score. The model uses a dataset of 500 individuals and applies feature engineering, feature selection, and scaling to predict the salary. It uses **Linear Regression** as the baseline model and **Ridge Regression** for improved performance.

## Project Components

1. **Data Generation**: A synthetic dataset is created with features including years of experience, education level, location, certifications, skill match score, and the target variable, salary.
2. **Feature Engineering**:
   - Encoding of categorical variables (education level and location).
   - Feature scaling using StandardScaler.
3. **Modeling**:
   - Feature selection using Recursive Feature Elimination (RFE).
   - Training the model using Linear Regression.
   - Hyperparameter tuning using Ridge Regression with GridSearchCV for improved accuracy.
4. **Evaluation**:
   - Model evaluation metrics like R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are provided.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

## Dataset

The dataset is generated synthetically with the following features:

- **Experience_Years**: Years of work experience (0-15 years).
- **Education_Level**: Educational qualification (Bachelors, Masters, PhD).
- **Location**: Location (New York, San Francisco, Austin, Berlin, Mumbai).
- **Certifications**: Number of certifications (0-4).
- **Skill_Match_Score**: A score (0.5 to 1.0) indicating how well a person's skills match a job description.
- **Salary**: Target variable (Salary in USD).

## Model Training & Evaluation

### 1. Data Preprocessing

- Label encoding for ordinal features (e.g., Education Level).
- One-hot encoding for nominal features (e.g., Location).
- Feature scaling using `StandardScaler`.
  
### 2. Feature Selection

- Used **Recursive Feature Elimination (RFE)** with a linear regression model to select the most important features.
  
### 3. Model Training

- A **Linear Regression** model is initially trained to predict the salary.
- Evaluated using R², MAE, and RMSE.

### 4. Hyperparameter Tuning

- Tuned **Ridge Regression** using **GridSearchCV** to find the optimal regularization parameter `alpha`.

### 5. Model Evaluation

- The final evaluation metrics on the test set are:
  - **R² Score**: 0.84
  - **Mean Absolute Error (MAE)**: 4142.88
  - **Root Mean Squared Error (RMSE)**: 5276.62

## Files

- **salary_model.pkl**: Trained linear regression model.
- **scaler.pkl**: Scaler used to standardize features.
- **selected_features.pkl**: List of selected features after RFE.
  
These files can be loaded and used to predict salary for new data.

## How to Use the Model

1. **Load the Pretrained Model and Scaler**:

```python
import joblib

model = joblib.load('salary_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')
```

2. **Prepare Your Data**: Make sure your input data contains the same features that were used to train the model (`Experience_Years`, `Education_Level`, `Certifications`, `Skill_Match_Score`, and one-hot encoded `Location`).

3. **Scale and Predict**:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example new data
new_data = pd.DataFrame({
    'Experience_Years': [5.0],
    'Education_Level': [1],  # Masters
    'Certifications': [3],
    'Skill_Match_Score': [0.85],
    'Location_New York': [1],  # Location as one-hot encoded
    'Location_Berlin': [0],
    'Location_Mumbai': [0],
    'Location_San Francisco': [0]
})

# Scale the input features
scaled_features = scaler.transform(new_data)

# Predict salary
predicted_salary = model.predict(scaled_features)
print(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
```

## Conclusion

This project provides a robust salary prediction model using multiple features, including experience, education, location, certifications, and skill match score. The model can be easily extended to incorporate additional features or other machine learning algorithms for more refined predictions.

