# Titanic Survival Prediction using K-Nearest Neighbors (KNN)

## Overview

This project uses the **Titanic dataset** to predict whether a passenger survived using the **K-Nearest Neighbors (KNN)** classification algorithm.

The dataset contains passenger information such as passenger class, sex, age, number of siblings/spouses, parents/children, fare, embarkation point, and whether the passenger was traveling alone.

## Dataset

The Titanic dataset was loaded using Seaborn:

```python
df = sns.load_dataset('titanic')
```

The original dataset contains **891 rows and 15 columns**. During preprocessing, unnecessary and redundant columns were removed, leaving 9 useful columns for the model.

## Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns such as `class`, `who`, `adult_male`, `deck`, `embark_town`, and `alive`.
- Filled missing values in the `age` column using the mean age.
- Removed rows with missing values in the `embarked` column.
- Converted categorical columns `sex` and `embarked` into numerical values using `LabelEncoder`.
- Converted the dataset columns to integer type.
- Separated the target variable `survived` from the input features.
- Split the data into training and testing sets using an **80:20 split**.
- Applied `StandardScaler` to scale the features before training KNN.

## Model Used

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning classification algorithm. It predicts the class of a new data point by looking at the classes of its nearest neighboring data points.

In this project:

```python
knn = KNeighborsClassifier(n_neighbors=5)
```

The model was trained using 5 nearest neighbors.

## Model Evaluation

The KNN model achieved the following accuracy:

**KNN Accuracy: 79.21%**

The confusion matrix was:

```text
[[89, 20],
 [17, 52]]
```

This means the model correctly classified 89 non-survivors and 52 survivors in the test set.

## Comparison with Logistic Regression

I also compared the KNN model with **Logistic Regression** on the same Titanic prediction task.

| Model | Accuracy |
|---|---:|
| Logistic Regression | **80%** |
| KNN | **79.21%** |

### Result

Logistic Regression performed slightly better than KNN on this dataset:

- Logistic Regression: **0.80**
- KNN: **0.7921 ≈ 0.79**

The difference is small, so both models performed reasonably well. In this particular experiment, **Logistic Regression gave the better accuracy**.

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Scikit-learn
- Google Colab

## Key Learning

Through this project, I practiced:

- Data cleaning and preprocessing
- Handling missing values
- Label encoding
- Feature scaling using `StandardScaler`
- Train-test splitting
- KNN classification
- Model evaluation using accuracy and confusion matrix
- Comparing different machine learning models

## Conclusion

The KNN classifier achieved an accuracy of **79.21%** on the Titanic test set. When compared with Logistic Regression, which achieved **80% accuracy**, Logistic Regression performed slightly better in this experiment.

This comparison helped demonstrate that different machine learning algorithms can produce different results on the same dataset, making model comparison an important part of the machine learning workflow.
