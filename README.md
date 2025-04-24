# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SELVA JOBIN S
RegisterNumber:  212223220102
```
```py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np

data = pd.read_csv("Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print("Predicted:", y_pred)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("New prediction:", dt.predict(np.array([[5, 6]])))
```
## Output:

![image](https://github.com/user-attachments/assets/8bba9fbd-6928-4642-939a-ee71c1376139)
![image](https://github.com/user-attachments/assets/257bf4a3-94af-42e9-849d-9616502f9664)
![image](https://github.com/user-attachments/assets/455fedb8-ffe2-49d9-9681-a5e264426c2f)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
