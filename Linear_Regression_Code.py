# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 19:14:17 2021

@author: ayana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
sal=pd.read_csv("C:\\Users\\ayana\\Downloads\\Salary_Data.csv") 
sal.columns # dataset name
sal.columns=['ye','s'] #rename the coulumn name of the dataset
#EDA
plt.hist(sal.ye)
plt.boxplot(sal.ye,vert=True)

plt.hist(sal.s)
plt.boxplot(sal.s,vert=False)
plt.scatter(x=sal.ye,y=sal.s,color="red");plt.xlabel("years of experience");plt.ylabel("salary") #scatter plot without line
#correlation
sal.ye.corr(sal.s) #0.978
np.corrcoef(sal.ye,sal.s)  

#base model
import statsmodels.formula.api as smf
model=smf.ols("s~ye",data=sal).fit() #linear regression model is fitted by taking data from dataset 
model.params #to get coefficients b0,b1

model.summary() #0.957
model.conf_int(0.05)
pred = model.predict(sal.ye)
pc = pred.corr(sal.s)  #0.9782

#scatterplot
import matplotlib.pylab as plt
plt.scatter(x=sal['ye'],y=sal['s'],color='red');plt.plot(sal['ye'],pred,color ="black");plt.xlabel('years of experience');plt.ylabel('salary')
#residual plot
error1 = pred-sal.s
R1 = model.resid_pearson
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(sal.s,pred)) #5592.043
#plot
plt.scatter(x=pred,y=sal.s,color = "green");plt.xlabel("prediction"),plt.ylabel("salary")
plt.plot(R1,'o');plt.axhline(y=0,color='green');plt.xlabel("observation no"),plt.ylabel("Normalised residual")
plt.scatter(x=pred,y=sal.s,color="pink")