# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # matrix of features(feature = independent variable)
y = dataset.iloc[:, 4].values # dependent variable vector

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:] # we are discarding 1st column(also the first column of dummy variables) to remove redundancies. Although dummy variable trap is removed by the python library itself but sometimes we need to do it manually

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)""" # since sc_X object is already fitted to training set, we just transform test set

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)  

# Predicting the Test set Results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # here if we had given value of arr to values and value of values to arr it would have added a column of ones at last. we use .astype to convert array of ones to an array of integers
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS - Ordinary least squares, fitting multiple linear model to X_opt matrix
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS - Ordinary least squares, fitting multiple linear model to X_opt matrix
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS - Ordinary least squares, fitting multiple linear model to X_opt matrix
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS - Ordinary least squares, fitting multiple linear model to X_opt matrix
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # OLS - Ordinary least squares, fitting multiple linear model to X_opt matrix
regressor_OLS.summary()

