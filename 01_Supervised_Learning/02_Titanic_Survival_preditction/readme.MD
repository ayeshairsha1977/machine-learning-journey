
# 🚢 Titanic Survival Prediction using Logistic Regression

This project uses the **Titanic dataset** to predict whether a passenger survived the Titanic disaster using **Logistic Regression**.

The project focuses on understanding the complete basic machine learning workflow, including data exploration, preprocessing, encoding categorical data, feature scaling, and preparing data for model training.

## 📊 Dataset

The Titanic dataset is loaded using Seaborn:

```python
df = sns.load_dataset("titanic")
```

The dataset contains **891 rows and 15 columns** initially.

Some important features include:

* `survived` - Target variable indicating whether the passenger survived
* `pclass` - Passenger class
* `sex` - Passenger gender
* `age` - Passenger age
* `sibsp` - Number of siblings/spouses aboard
* `parch` - Number of parents/children aboard
* `fare` - Passenger ticket fare
* `embarked` - Port of embarkation
* `alone` - Whether the passenger was traveling alone

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Removing unnecessary columns

Columns such as `deck`, `class`, `who`, `adult_male`, `embark_town`, and `alive` were removed.

### 2. Handling missing values

Missing values in `age` were replaced with the mean age:

```python
df["age"].fillna(df["age"].mean(), inplace=True)
```

Missing values in `embarked` were removed using:

```python
df.dropna(subset=["embarked"], inplace=True)
```

After this, the dataset contained **889 rows**.

### 3. Encoding categorical variables

`LabelEncoder` was used to convert the categorical columns `sex` and `embarked` into numerical values.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"])
```

The dataframe was then converted to integer type.

### 4. Feature Scaling

`StandardScaler` was used to standardize:

* `pclass`
* `age`
* `fare`
* `embarked`

```python
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

df[["pclass", "age", "fare", "embarked"]] = std.fit_transform(
    df[["pclass", "age", "fare", "embarked"]]
)
```

This transformed the selected features to a standardized scale.

## 🎯 Features and Target

The target variable is:

```python
y = df["survived"]
```

The remaining columns are used as input features:

```python
X = df.drop("survived", axis=1)
```

So:

```text
X → Passenger features
y → Survival status
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn
* Google Colab

## 🧠 Concepts Practiced

* Data loading
* Data exploration
* Handling missing values
* Feature selection
* Categorical encoding
* Label Encoding
* Feature scaling
* Standardization
* Feature-target separation
* Logistic Regression preparation

## 📁 Project Structure

```text
Titanic-Survival-Logistic-Regression/
│
├── Titanic_Survival_logesitc_regression.ipynb
└── README.md
```

## 🚀 Learning Goal

The main goal of this project was to understand how a real dataset is prepared before applying a machine learning model and to practice the preprocessing steps required for a classification problem.

---

**Model:** Logistic Regression
**Dataset:** Titanic
**Status:** Completed preprocessing and model preparation
