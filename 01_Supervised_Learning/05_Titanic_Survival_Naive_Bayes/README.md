# Titanic Survival Prediction

## Overview
This project uses the Titanic dataset to predict whether a passenger survived using three machine learning classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

The goal is to compare the performance of these models on the same prepared dataset.

## Dataset
The Titanic dataset is loaded using Seaborn:

```python
df = sns.load_dataset('titanic')
```

The target variable is:

- `survived` → 0 = did not survive, 1 = survived

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed the `deck` column because it contained missing values.
2. Removed redundant columns:
   - `alive`
   - `class`
   - `who`
   - `adult_male`
   - `embark_town`
3. Filled missing values in `age` using the mean age.
4. Removed rows with missing `embarked` values.
5. Converted categorical columns `sex` and `embarked` into numerical values using `LabelEncoder`.
6. Converted the resulting dataframe to integer type.
7. Separated features (`X`) and target (`y`).
8. Split the data into training and testing sets using an 80/20 split with `random_state=42`.

## Models Compared

### 1. Logistic Regression
Logistic Regression was used as a classification model to predict the probability of survival and classify passengers into survived/not survived.

**Accuracy: 81%**

### 2. K-Nearest Neighbors (KNN)
KNN predicts the class of a passenger based on the classes of nearby training examples.

**Accuracy: 79%**

### 3. Gaussian Naive Bayes
Gaussian Naive Bayes was used as another classification approach.

**Accuracy: 77%**

## Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 81% |
| KNN | 79% |
| Gaussian Naive Bayes | 77% |

### Best Performing Model

Based on accuracy, **Logistic Regression performed best with 81% accuracy**, followed by KNN at 79% and Gaussian Naive Bayes at 77%.

## Evaluation

The models were evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

## Conclusion

Among the three models tested, Logistic Regression achieved the highest accuracy on the test set at 81%. KNN achieved 79%, while Gaussian Naive Bayes achieved 77%.

This project provides a basic comparison of different classification algorithms on the Titanic survival prediction problem.
