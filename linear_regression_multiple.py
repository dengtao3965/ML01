import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_table("data/multiple_variable.txt",header=None,sep='\t',quoting=3)

tx=[]
ty=[]
total=[]
for i in range(len(data[0])):
    tx.append([data[0][i],data[1][i],data[2][i]])
    ty.append([data[3][i]])

tx=np.array(tx)
ty=np.array(ty)

x=tf.placeholder("float",[None,3])
y=tf.placeholder("float",[None,1])

w=tf.Variable(tf.random_uniform([3,1],-1.,1.))
b=tf.Variable(tf.zeros([1]))

py=tf.matmul(x,w)+b
loss = tf.reduce_sum(tf.pow(py - y, 2))

optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

session=tf.Session()
init=tf.global_variables_initializer()
session.run(init)
writer=tf.summary.FileWriter('graphs',session.graph)

print("w=",session.run(w))
print("b=",session.run(b))

for i in range(1000):
    session.run(optimizer,{x:tx,y:ty})
    if i%10==0:
        l=session.run(loss,{x:tx,y:ty})
        print("w=",session.run(w),"  b=",session.run(b),"  loss=",l)
        total.append(l)

print("-----end------")
print("loss=",session.run(loss,{x:tx,y:ty}))
print("w=",session.run(w))
print("b=",session.run(b))

print("py=",session.run(py,{x:[[0.25,0.25,0.25]]}))
plt.plot(total)
plt.show()