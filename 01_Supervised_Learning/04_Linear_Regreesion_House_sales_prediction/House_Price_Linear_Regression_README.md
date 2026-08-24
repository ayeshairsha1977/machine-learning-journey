# House Price Prediction using Linear Regression

## Overview

This project predicts house prices using **Linear Regression** on the King County house sales dataset.

Workflow:
**Data Loading → Data Cleaning → EDA → Correlation Analysis → Feature Selection → Train-Test Split → Linear Regression → Prediction → Evaluation**

## Dataset

- **Rows:** 21,613
- **Columns:** 21
- **Target:** `price`

The dataset contains information such as bedrooms, bathrooms, living area, lot size, grade, view, location, and other house characteristics.

## Data Cleaning

- Checked the dataset using `df.info()`
- Checked for duplicate rows
- Checked for missing values
- No duplicate rows were found
- No missing values were found
- Converted `date` from `object` to Pandas `datetime64[ns]`

## Correlation Analysis

The strongest correlations with `price` were:

| Feature | Correlation |
|---|---:|
| `sqft_living` | 0.7020 |
| `grade` | 0.6674 |
| `sqft_above` | 0.6056 |
| `sqft_living15` | 0.5854 |
| `bathrooms` | 0.5251 |
| `view` | 0.3973 |
| `sqft_basement` | 0.3238 |
| `bedrooms` | 0.3084 |
| `lat` | 0.3070 |

Some living-area features were highly correlated with each other:

- `sqft_living` ↔ `sqft_above` = **0.8766**
- `sqft_living` ↔ `sqft_living15` = **0.7564**
- `sqft_above` ↔ `sqft_living15` = **0.7319**

Because these features contain overlapping information, `sqft_living` was selected as the main living-area feature for the initial model.

## Selected Features

```python
features = [
    'sqft_living',
    'grade',
    'bathrooms',
    'view',
    'sqft_basement',
    'bedrooms',
    'lat'
]

X = df[features]
y = df['price']
```

## Train-Test Split

An **80:20 train-test split** was used with `random_state=42`.

## Model

**Linear Regression**

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

## Prediction

```python
y_pred = model.predict(X_test)
```

## Model Evaluation

Current results:

- **R² Score:** 0.6403
- **Adjusted R² Score:** 0.6397

The R² score indicates that the selected features explain approximately **64% of the variation in house prices** on the test set.

## Future Improvements

- Add MAE, MSE, and RMSE
- Compare different feature combinations
- Test the effect of adding/removing correlated features
- Add actual-vs-predicted visualizations
- Experiment with other regression algorithms

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Google Colab

## Key Learnings

- Data cleaning
- Missing-value and duplicate checking
- Date conversion
- Correlation analysis
- Feature selection
- Understanding multicollinearity
- Train-test splitting
- Linear Regression
- Making predictions
- R² and Adjusted R² evaluation

## Project Status

**In Progress**

The basic Linear Regression model, prediction workflow, and initial evaluation are completed. Additional metrics and visualizations can be added later.
