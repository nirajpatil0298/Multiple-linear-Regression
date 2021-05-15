# -*- coding: utf-8 -*-
"""
Created on Sun May  9 08:35:24 2021

@author: Admin
"""
# Multiple Linear Regression

#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#Encoding Categorical data
#Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:]

#Splitting the dataset into Training set and testing set
from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.2 ,random_state = 0)

#Feature Scaaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train , y_train)

#Predicting the Test set Result
y_pred = regressor.predict(X_test)

#bulding the optimal model using backword elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

#Fit the Model with all possible predictors
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#there X2 has the Highest p-value then go for step 4
#step 4 is Remove the predictor which has high p-value

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#there X1 has the Highest p-value then go for step 4
#step 4 is Remove the predictor which has high p-value


X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#there X2(Admin column) has the Highest p-value then go for step 4
#step 4 is Remove the predictor which has high p-value


X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#there X2 (Marketing Spend column) has the Highest p-value then go for step 4
#step 4 is Remove the predictor which has high p-value


X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Therefore Only One independent variable R&D spend  which definentlys makes very 
#power full predictor

