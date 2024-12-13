# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Step 1: Read the employee data from a CSV file.
Step 2: Check for null values and encode categorical variables.
Step 3: Define the features (X) and target (y).
Step 4: Split the data into training and testing sets.
Step 5: Train the Decision Tree Classifier.
step 6: Make predictions on the test data.
step 7: Calculate the accuracy of the model.
step 8: Use the model to predict new data.
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MARINO SARISHA T
RegisterNumber:212223240084  
*/
import pandas as pd
data=pd.read_csv( "Employee.csv" )
data . info( )
data.isnull().sum()
data ["left"].value_counts ( )

from sklearn. preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
y=data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x, y , test_size=0.2 ,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("Predict:",dt.predict([[0.5,0.8,9,260,6,0,1,2]]))
```

## Output:
![Screenshot 2024-11-17 144223](https://github.com/user-attachments/assets/155b6d87-aa0d-4026-a3de-06c255ba2998)
![Screenshot 2024-11-17 144240](https://github.com/user-attachments/assets/7f97d59b-a051-48d7-bf99-d93ae7a8efa2)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
