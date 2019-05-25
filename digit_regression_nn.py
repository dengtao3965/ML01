import numpy as np
import pandas as pd
from keras.utils import np_utils
import os
import tensorflow as tf


def load_train_data(file_name, normalize=True):
    x_train, y_train = [], []
    with open(file_name) as my_file:
        header = my_file.readline()
        for line in my_file.readlines():
            line = line.strip().split(',')
            x_train.append(line[1:])
            y_train.append(int(line[0]))

    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train)

    if normalize == True:
        x_train /= 255

    return x_train, y_train

def load_test_data(file_name, normalize=True):
    x_test = []
    with open(file_name) as my_file:
        hader = my_file.readline()
        for line in my_file.readlines():
            line = line.strip().split(',')
            x_test.append(line)

    x_test = np.array(x_test).astype('float32')
    if normalize == True:
        x_test /= 255

    return x_test


def save_result(y_pred, file_name):
    result_df = pd.DataFrame({'ImageId': range(1, len(y_pred) + 1), 'Label': y_pred})
    result_df.to_csv(file_name, index=False)

x_train,y_train=load_train_data("data/train.csv")
x_test=load_test_data("data/test.csv")
y_train=np_utils.to_categorical(y_train)

batch_size=128
batch_num=int(len(x_train)/batch_size)

x=tf.placeholder("float",[None,784])
y=tf.placeholder("float",[None,10])
#hidden_layer_1
w_1=tf.Variable(tf.random_normal([784,256]))
b_1=tf.Variable(tf.random_normal([256]))
hidden_1=tf.nn.sigmoid(tf.add(tf.matmul(x,w_1),b_1))
#hidden_layer_2
w_2=tf.Variable(tf.random_normal([256,100]))
b_2=tf.Variable(tf.random_normal([100]))
hidden_2=tf.nn.sigmoid(tf.add(tf.matmul(hidden_1,w_2),b_2))
#hidden_layer_3
w_3=tf.Variable(tf.random_normal([100,64]))
b_3=tf.Variable(tf.random_normal([64]))
hidden_3=tf.nn.sigmoid(tf.add(tf.matmul(hidden_2,w_3),b_3))
#output_layer
out_w=tf.Variable(tf.random_normal([64,10]))
out_b=tf.Variable(tf.random_normal([10]))
out_y=tf.add(tf.matmul(hidden_3,out_w),out_b)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_y,labels=y))
optimizer=tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(150):
        avg_cost=0.

        for i in range(batch_num):
            batch_xs=x_train[i*batch_size:(i+1)*batch_size]
            batch_xy=y_train[i*batch_size:(i+1)*batch_size]

            sess.run(optimizer,feed_dict={x:batch_xs,y:batch_xy})
            avg_cost=sess.run(loss,feed_dict={x:batch_xs,y:batch_xy})

        print('epoch:%d,cost:%.9f'%(epoch+1,avg_cost))

    y_p=sess.run(out_y,{x:x_test})
    y_p=np.argmax(y_p,axis=-1)
    save_file = os.path.join('result', 'nn.csv')
    save_result(y_p,save_file)