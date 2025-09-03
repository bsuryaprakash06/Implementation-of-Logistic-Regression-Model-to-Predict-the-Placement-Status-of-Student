# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

**Name:** B Surya Prakash

**Reg No:** 212224230281
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
# Algorithm for Placement Prediction Using Logistic Regression

**Step 1: Import required libraries**  
- Import pandas for data handling.  
- Import scikit-learn modules (`LabelEncoder`, `train_test_split`, `LogisticRegression`, `accuracy_score`, `confusion_matrix`, `classification_report`).  

**Step 2: Load dataset**  
- Read `Placement_Data.csv` into a DataFrame using `pd.read_csv()`.  

**Step 3: Preprocess dataset**  
- Create a copy of the dataset.  
- Drop unnecessary columns (`sl_no`, `salary`).  

**Step 4: Check data quality**  
- Check for missing values using `isnull().sum()`.  
- Check for duplicate rows using `duplicated().sum()`.  

**Step 5: Encode categorical variables**  
- Apply `LabelEncoder` to categorical columns:  
  (`gender`, `ssc_b`, `hsc_b`, `hsc_s`, `degree_t`, `workex`, `specialisation`, `status`).  

**Step 6: Split features and target**  
- Define `X` as all input features (all columns except `status`).  
- Define `y` as the target variable (`status`).  

**Step 7: Split dataset into training and testing sets**  
- Use `train_test_split` with `test_size=0.2` and `random_state=0`.  

**Step 8: Train the Logistic Regression model**  
- Initialize `LogisticRegression(max_iter=1000, random_state=0)`.  
- Fit the model using `X_train` and `y_train`.  

**Step 9: Make predictions on test data**  
- Use the trained model to predict on `X_test` → `y_pred`.  

**Step 10: Evaluate the model**  
- Calculate accuracy using `accuracy_score(y_test, y_pred)`.  
- Generate a confusion matrix using `confusion_matrix(y_test, y_pred)`.  
- Generate a classification report using `classification_report(y_test, y_pred)`.  

**Step 11: Predict for a new sample**  
- Prepare a single row of data in the same order as training features.  
- Use `lr.predict([[...values...]])` or create a DataFrame with matching column names to avoid warnings.  

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: B Surya Prakash
RegisterNumber:  212224230281
*/
```
```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
data.head()
```
```python
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
data1.head()
```
```python
data1.isnull().sum()
```
```python
data1.duplicated().sum()
```
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
data1
```

```python
x=data1.iloc[:,:-1]
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
x
```

```python
y=data1["status"]
y
```
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```

```python
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```python
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```

```python
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
```python
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
<img width="1004" height="233" alt="image" src="https://github.com/user-attachments/assets/e54cbbca-9bc8-42c9-8a36-e94810e975e7" />

<img width="995" height="233" alt="image" src="https://github.com/user-attachments/assets/208bdf9b-83cf-44f2-8f51-07b571a9c09a" />

<img width="989" height="250" alt="image" src="https://github.com/user-attachments/assets/be18b901-304f-45d4-9e45-3329c3cd8753" />

<img width="999" height="60" alt="image" src="https://github.com/user-attachments/assets/ae81a450-f193-4945-bc2f-7222ae763b3f" />

<img width="995" height="389" alt="image" src="https://github.com/user-attachments/assets/b4083e41-4595-47fd-b8d5-849170a53ee8" />

<img width="996" height="380" alt="image" src="https://github.com/user-attachments/assets/edac7b90-a95d-4284-ba2b-a64e28f42858" />

<img width="1000" height="207" alt="image" src="https://github.com/user-attachments/assets/39ba5fb3-c256-429c-b27a-c424611e5165" />

**Y PREDICTION:**
<img width="995" height="94" alt="image" src="https://github.com/user-attachments/assets/03eafdb0-0809-4fba-9f04-8eaa84b9ceb2" />

**ACCURACY:**
<img width="994" height="36" alt="image" src="https://github.com/user-attachments/assets/2498f747-2cb3-450c-a432-a5503ec34188" />

**CONFUSION:**
<img width="1002" height="52" alt="image" src="https://github.com/user-attachments/assets/7af6ed58-39d9-4219-bf09-b82a113b2f74" />

**Classification_report:**
<img width="998" height="152" alt="image" src="https://github.com/user-attachments/assets/b0253302-3c73-4277-afd2-0771fe2a31b5" />

<img width="998" height="32" alt="image" src="https://github.com/user-attachments/assets/ff2b5198-ea61-43ef-9008-421564f6e0fc" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
