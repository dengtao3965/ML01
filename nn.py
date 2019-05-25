import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist_data=input_data.read_data_sets("data",one_hot=True)
batch_size=75
batch_num=mnist_data.train.num_examples//batch_size
print(batch_num)

tx=tf.placeholder(tf.float32,[None,784])
ty=tf.placeholder(tf.float32,[None,10])

w_1=tf.Variable(tf.zeros([784,150]))
b_1=tf.Variable(tf.random_normal([150]))
L1=tf.nn.relu(tf.matmul(tx,w_1)+b_1)

w_2=tf.Variable(tf.zeros([150,10]))
b_2=tf.Variable(tf.random_normal([10]))

prediction=tf.nn.softmax(tf.matmul(L1,w_2)+b_2)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ty,logits=prediction))
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
correct_pred=tf.equal(tf.arg_max(ty,1),tf.arg_max(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
with tf.Session() as sses:
    sses.run(init)
    for i in range(51):
        for batch in range(batch_num):
            batch_x,batch_y=mnist_data.train.next_batch(batch_size)
            sses.run(train_step,feed_dict={tx:batch_x,ty:batch_y})

        acc=sses.run(accuracy,feed_dict={tx:mnist_data.test.images,ty:mnist_data.test.labels})
        print("Step "+str(i)+",Training Accuracy "+str(acc))

print("end!")   