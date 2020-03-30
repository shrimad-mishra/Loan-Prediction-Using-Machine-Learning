# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the required data file using pandas

# Load dataset
url="https://raw.githubusercontent.com/callxpert/datasets/master/Loan-applicant-details.csv"
names=['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
data=pd.read_csv(url, names=names)

# Dummies
from sklearn.preprocessing import LabelEncoder
var_mod=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le=LabelEncoder()
for i in var_mod:
    data[i]=le.fit_transform(data[i])

# Deciding the x and y
x=data.iloc[:, [6,7,8,9,10]].values
y=data.iloc[:,-1].values

# Now splitting of the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

# Now Deciding the best model for the evaluation 
from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(x_train,y_train)

# Now predicting the results
y_pred=regressor.predict(x_test)

# Analysing the results
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))