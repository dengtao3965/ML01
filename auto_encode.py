import numpy as np
import pandas as pd
from keras.utils import np_utils
import os
import tensorflow as tf
import matplotlib.pyplot as plt

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
ae_train=np.concatenate((x_train,x_test),axis=0)
hidden_units=64
input_units=ae_train.shape[1]

inputs_=tf.placeholder(tf.float32,(None,input_units),name='inputs_')
targets_=tf.placeholder(tf.float32,(None,input_units),name='targets_')

hidden_=tf.layers.dense(inputs_,hidden_units,activation=tf.nn.relu)

logits_=tf.layers.dense(hidden_,input_units,activation=None)

outputs_=tf.sigmoid(logits_,name='outputs_')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
cost = tf.reduce_mean(loss)

learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs=20

for e in range(epochs):
    for idx in range(len(ae_train)//batch_size):
        batch = ae_train[idx*batch_size:(idx+1)*batch_size]

        batch_cost, _ = sess.run([cost, optimizer],
                                 feed_dict={inputs_: batch,
                                            targets_: batch})
        print("Epoch: {}/{}".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))



reconstructed, x_train_compressed = sess.run([outputs_, hidden_],
                                     feed_dict={inputs_: x_train})

reconstructed, x_test_compressed = sess.run([outputs_, hidden_],
                                     feed_dict={inputs_: x_test})

from sklearn.svm import SVC
clf = SVC()
save_file = os.path.join('result', 'svm.csv')

clf.fit(x_train_compressed, y_train)
y_pred = clf.predict(x_test_compressed)
save_result(y_pred, save_file)

sess.close()
