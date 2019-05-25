
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


data=pd.read_table("data/one_variable.txt",header=None,sep='\t',quoting=3)
tx=np.array(data[0])
ty=np.array(data[1])

tx=[]
for index in range(len(data)):
    tx.append([1.,data[0][index]])

tx=np.array(tx)
ty=np.array(data[1])

lr=LinearRegression()
lr.fit(tx,ty)

print(lr.coef_)
print(lr.intercept_)
print(lr.predict([[1.,0.25]]))

