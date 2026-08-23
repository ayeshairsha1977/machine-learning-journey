
# 🚢 Titanic Survival Prediction using Logistic Regression

## 📌 Project Overview

This project uses the **Titanic dataset** to prepare passenger data for a **Logistic Regression classification model**.

The dataset initially contains **891 rows and 15 columns**. The main objective is to understand how real-world data is cleaned, transformed, and prepared before applying a machine learning model.

## 📊 Dataset Features

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

---

## 🧹 Data Preprocessing

### 1. Removing Unnecessary Columns

The following columns were removed because they were redundant or not required for this project:

* `deck`
* `class`
* `who`
* `adult_male`
* `embark_town`
* `alive`

### 2. Handling Missing Values

The `age` column contained missing values. These were replaced with the **mean age**:

```python
df["age"].fillna(df["age"].mean(), inplace=True)
```

Missing values in the `embarked` column were removed:

```python
df.dropna(subset=["embarked"], inplace=True)
```

After removing rows with missing `embarked` values, the dataset contained **889 rows**.

### 3. Encoding Categorical Variables

The categorical columns `sex` and `embarked` were converted into numerical values using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"])
```

This allowed the categorical features to be used by the machine learning model.

### 4. Feature Scaling

`StandardScaler` was used to standardize the following numerical features:

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

Standardization transforms the selected features so that they are on a comparable scale.

---

## 🎯 Features and Target

The target variable is `survived`:

```python
y = df["survived"]
```

The remaining columns are used as input features:

```python
X = df.drop("survived", axis=1)
```

Therefore:

```text
X → Passenger features
y → Survival status
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn
* Google Colab

---

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
* Classification problem preprocessing

---

## 📁 Project Structure

```text
Titanic-Survival-Logistic-Regression/
│
├── Titanic_Survival_logistic_regression.ipynb
└── README.md
```

---

## 🚀 Learning Goal

The main goal of this project was to understand how a real-world dataset is prepared before applying a machine learning model.

Through this project, the following preprocessing workflow was practiced:

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Remove Unnecessary Columns
     ↓
Handle Missing Values
     ↓
Encode Categorical Features
     ↓
Feature Scaling
     ↓
Separate Features and Target
     ↓
Prepare Data for Logistic Regression
```

## 🤖 Model

**Logistic Regression**

## 📂 Dataset

**Titanic Dataset**

## ✅ Status

**Completed preprocessing and model preparation.**
