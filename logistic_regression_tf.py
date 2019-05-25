import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

total=[]
rate=0.0075
data=pd.read_table("data/ex2data1.txt",header=None,sep=',',quoting=3)

tx=np.array(data.iloc[:,0:2])
ty=np.array(data.iloc[:,2:3])

scaler=StandardScaler()
scaler.fit(tx)
tx=scaler.transform(tx)

x=tf.placeholder("float",[None,2])
y=tf.placeholder("float",[None,1])

w=tf.Variable(tf.random_uniform([2,1],-1.,1.))
b=tf.Variable(tf.zeros([1]))

py=tf.nn.sigmoid(tf.matmul(x,w)+b)
loss = tf.reduce_mean(- y * tf.log(py) - (1 - y) * tf.log(1 - py))

optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

init = tf.global_variables_initializer()

session=tf.Session()
session.run(init)

print("w=",session.run(w))
print("b=",session.run(b))

for i in range(15000):
    session.run(optimizer,{x:tx,y:ty})
    if i%500==0:
        l=session.run(loss,{x:tx,y:ty})
        print("w=",session.run(w),"  b=",session.run(b),"  loss=",l)
        total.append(l)

print("-----end------")
print("loss=",session.run(loss,{x:tx,y:ty}))
print("w=",session.run(w))
print("b=",session.run(b))

print("py=",session.run(py,{x:[[0.25,0.25]]}))
plt.plot(total)
plt.show()




