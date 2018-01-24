import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
learn_rate=0.01
batch_size=128
n_epochs=30
mnist=input_data.read_data_sets('data/mnist',one_hot=True)
X=tf.placeholder(tf.float32,[batch_size,784],name='x')
Y=tf.placeholder(tf.int32,[batch_size,10],name='y')
w=tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name='weights')
b=tf.Variable(tf.zeros([1,10]),name='bias')
logits=tf.matmul(X,w)+b
entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y,name='loss')
loss=tf.reduce_mean(entropy)
optimizer=tf.train.AdamOptimizer(learn_rate).minimize(loss)
with tf.Session()as  sess:
    writer=tf.summary.FileWriter('./graphs/mnist',sess.graph)
    start_time=time.time()
    sess.run(tf.global_variables_initializer())
    n_batch=int(mnist.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss=0
        for _ in range(n_batch):
            x_batch,y_batch=mnist.train.next_batch(batch_size)
            _,loss_bacth= sess.run([optimizer,loss],feed_dict={X:x_batch,Y:y_batch})
            total_loss+=loss_bacth
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batch))

    print('Total time: {0} seconds'.format(time.time() - start_time))
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # need numpy.count_nonzero(boolarr) :(

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run(accuracy, feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds+=accuracy_batch

    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))

    writer.close()




