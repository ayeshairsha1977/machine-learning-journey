# Titanic Survival Prediction using Decision Tree

## Overview

This project uses the Titanic dataset to predict whether a passenger survived or not using a Decision Tree Classifier.

The dataset contains information such as passenger class, sex, age, number of siblings/spouses, parents/children, fare, port of embarkation, and whether the passenger was traveling alone.

## Dataset

The Titanic dataset was loaded using Seaborn's built-in dataset.

Features used:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked
- Alone

Target variable:
- `Survived`

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed the `deck` column because it contained many missing values.
2. Filled missing `age` values with the mean age.
3. Removed rows with missing `embarked` values.
4. Converted categorical features such as `sex` and `embarked` into numerical values using `LabelEncoder`.
5. Separated the features (`X`) and target (`y`).
6. Split the dataset into training and testing sets using an 80/20 split.

## Model

A Decision Tree Classifier was used with:

```python
model = DecisionTreeClassifier(random_state=42)
```

The model was trained on the training dataset and evaluated on the test dataset.

## Results

The Decision Tree achieved a test accuracy of approximately:

**74.72%**

The classification report showed:

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| 0 | 0.83 | 0.73 | 0.78 |
| 1 | 0.65 | 0.77 | 0.70 |

Overall:
- **Test Accuracy:** 74.72%
- **Test Samples:** 178

Confusion matrix:

```text
[[80 29]
 [16 53]]
```

## Conclusion

The Decision Tree achieved approximately 74.72% accuracy on the processed Titanic test dataset.

Decision Trees can work well when features are categorical because they can make decisions through feature-based splits after appropriate encoding. However, having categorical features alone does not guarantee that a Decision Tree will be the best model. The best model should be selected by comparing different algorithms using the same evaluation procedure.

## Key Learning

This project demonstrates:

- Data cleaning
- Handling missing values
- Categorical encoding
- Train-test splitting
- Decision Tree classification
- Model evaluation
- Accuracy, precision, recall, and F1-score
- Confusion matrix

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Scikit-learn
- Matplotlib
