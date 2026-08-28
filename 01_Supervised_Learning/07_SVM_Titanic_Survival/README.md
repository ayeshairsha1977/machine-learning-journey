# Titanic Survival Prediction using SVM

## Overview

This project uses the Titanic dataset to predict whether a passenger survived or not using a Support Vector Machine (SVM) classifier.

The SVM model uses the **RBF (Radial Basis Function) kernel** to learn a non-linear decision boundary between the classes.

## Dataset

The Titanic dataset was loaded using Seaborn's built-in dataset:

```python
df = sns.load_dataset('titanic')
```

The following columns were removed because they were redundant or not used in the model:

- `class`
- `who`
- `adult_male`
- `embark_town`
- `alive`
- `deck`

The target variable is:

- `survived`

The remaining features are used to predict passenger survival.

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the Titanic dataset.
2. Removed unnecessary and redundant columns.
3. Filled missing `age` values using the mean age.
4. Removed rows where `embarked` was missing.
5. Converted the categorical `sex` feature into numerical values:
   - `male → 0`
   - `female → 1`
6. Converted the categorical `embarked` feature into numerical values:
   - `C → 0`
   - `Q → 1`
   - `S → 2`
7. Separated the features (`X`) and target (`y`).
8. Split the dataset into training and testing sets using an 80/20 split.

## Model

A Support Vector Classifier was used with the RBF kernel:

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf')
```

The model was trained using the training data and then used to predict survival on the test data.

## Results

The model was evaluated using **accuracy score** on the test set.

The notebook calculates the final accuracy using:

```python
accuracy_score(y_test, y_pred)
```

The exact numerical accuracy is not included in the notebook source, so it is intentionally not stated here rather than inventing a number.

## Why SVM?

SVM attempts to find a decision boundary that separates different classes. The RBF kernel allows SVM to model non-linear relationships between the features and the target.

SVM can be useful when the classes cannot be separated effectively using a simple linear boundary.

## Important Note About Feature Scaling

SVM models are generally sensitive to the scale of input features because the model relies on distances and the geometry of the feature space.

A standard approach is to scale the features **after splitting the data**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler should be fitted only on the training data and then used to transform the test data. This avoids data leakage.

The current notebook does not apply `StandardScaler`, so the reported model is based on the preprocessing actually present in the notebook.

## Key Learning

This project demonstrates:

- Loading a dataset using Seaborn
- Data cleaning
- Handling missing values
- Encoding categorical variables
- Train-test splitting
- Support Vector Machine classification
- RBF kernel
- Model prediction
- Accuracy evaluation
- The importance of feature scaling for SVM

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## Conclusion

The project demonstrates how an SVM with an RBF kernel can be applied to a binary classification problem such as Titanic survival prediction.

For a stronger SVM implementation, feature scaling should be added before training because the current notebook does not scale the input features.
