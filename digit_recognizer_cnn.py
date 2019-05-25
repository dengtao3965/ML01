import numpy as np
import pandas as pd
from keras.utils import np_utils
import os
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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

num_classes=10

train_file = os.path.join('data', 'train.csv')
test_file = os.path.join('data', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)

y_train = np_utils.to_categorical(y_train)

x = tf.placeholder("float", shape=[None, 784])  # 784＝28x28
y_ = tf.placeholder("float", shape=[None, num_classes])

W_conv1 = weight_variable([3, 3, 1, 32])  # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32

h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

W_fc1 = weight_variable([14 * 14 * 32, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, num_classes])
b_fc2 = bias_variable([num_classes])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


loss = -tf.reduce_sum(y_ * tf.log(y_conv))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


batch_size=100
batch_num=int(len(x_train)/batch_size)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(25):
        for idx in range(len(x_train)//batch_size):
            x_batch = x_train[idx*batch_size:(idx+1)*batch_size]
            y_batch = y_train[idx*batch_size:(idx+1)*batch_size]

            if idx % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: x_batch, y_: y_batch, keep_prob: 1.0})
                print("epoch %d, batch: %d, training accuracy %g" % (epoch, idx, train_accuracy))
            optimizer.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

    y_pred = []
    for idx in range(len(x_test) // batch_size):
        x_test_batch = x_test[idx * batch_size:(idx + 1) * batch_size]
        y_pred_batch = sess.run(y_conv, {x: x_test_batch, keep_prob: 1.0})
        y_pred_batch = np.argmax(y_pred_batch, axis=-1)
        for j in range(len(y_pred_batch)):
            y_pred.append(y_pred_batch[j])

    save_file = os.path.join('result', 'cnn_tensorflow.csv')
    save_result(y_pred, save_file)

    sess.close()
