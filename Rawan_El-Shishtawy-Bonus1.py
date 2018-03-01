# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:02:33 2018

@author: Rawan
"""



import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

le= LabelEncoder()
linReg= LinearRegression()

#reading the files
domain= pd.read_csv(r"E:\3rd 7asebat\Term2\AI\Sec_Notes\Bonus\Domain.csv",header=None, sep=":")

data= pd.read_csv(r"E:\3rd 7asebat\Term2\AI\Sec_Notes\Bonus\Data.csv",header=None,names=header_list)

header_list =  domain.iloc[:,0]
#remove cat. values
le.fit(data.iloc[:,0])
data.iloc[:,0]=le.transform(data.iloc[:,0])
data.iloc[:,0].head(7)

x=data.iloc[:,0:data.shape[1]-1]
y=data.rings
x_train, x_test, y_train, y_test = train_test_split( x,y, test_size=.33, random_state=42)


#model3 - Random Forest

model = RandomForestRegressor(n_estimators = 100,  n_jobs = -1,random_state =42, 
                              max_features = 3, min_samples_leaf =7,bootstrap= True,
                              max_depth=70, min_samples_split= 7)
                              

model.fit(x_train, y_train)
output= model.predict(x_test)


print( np.sqrt( mean_squared_error(y_test, output)))















