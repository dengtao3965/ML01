import pandas as pd
import tensorflow as tf
import numpy as np

data=pd.read_table("data/one_variable.txt",header=None,sep='\t',quoting=3)
tx=np.array(data[0])
ty=np.array(data[1])

x=tf.placeholder("float")
y=tf.placeholder("float")

w=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))

py=tf.add(tf.multiply(x, w), b)
loss = tf.reduce_sum(tf.pow(py - y, 2))

optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

session=tf.Session()
init=tf.global_variables_initializer()
session.run(init)
print("w=",session.run(w))
print("b=",session.run(b))

for i in range(1000):
    session.run(optimizer,{x:tx,y:ty})
    if i%10==0:
        print("w=",session.run(w),"  b=",session.run(b),"  loss=",session.run(loss,{x:tx,y:ty}))

print("-----end------")
print("loss=",session.run(loss,{x:tx,y:ty}))
print("w=",session.run(w))
print("b=",session.run(b))
print("py=",session.run(py,{x:[0.25]}))