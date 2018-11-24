# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:35:28 2018

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting polynomial regression to the model 

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#visualising the Linear Regression results
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title("Trurth or Bluff(Linear Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
#visualising the Polynomial Regression results

plt.scatter(x,y,color='green')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='black')
plt.title("Trurth or Bluff(Polynomial Regression)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
#visulaise the truth by your own choice using linear regression
print("Enter the positional level of which you want to know the correct result")
t=lin_reg.predict(float(input()))
print (t)
#visulaise the truth by your own choice using Polynomial regression
print("Enter the positional level of which you want to know the correct result")
y=lin_reg2.predict(poly_reg.fit_transform(float(input())))
print (y)




