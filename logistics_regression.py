import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import confusion_matrix


dataset=pd.read_csv("data/Social_NetWork_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

classifier=LogisticRegression()
classifier.fit(x_train,y_train)

py=classifier.predict(x_test)
cm=confusion_matrix(y_test,py)
# print("x_train:",x_train)
# print("x_test:",x_test)
print("py:",py)
print("y_test:",y_test)

err=0;
for i in range(len(py)):
    if py[i]!=y_test[i]:
        err=err+1

print(err/len(py))