import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
a=tf.constant([[1,2,3],[1,2,3],[1,3,5]])
b=tf.constant([[1,1,1],[2,2,2],[3,3,3]])
c=tf.ones([1,2,1,3,1,1])
with tf.Session()as sess:
    print(tf.argmax(a, 1).eval())
    print(tf.argmax(a, 0).eval())
    print(tf.reduce_sum(a,0).eval())
    print(tf.reduce_sum(a,1).eval())
    # print(tf.reduce_sum(a).eval())
    # print(tf.concat([a, b], 0).eval())
    # print(tf.concat([a, b], 1).eval())
    # print(c.eval())
    # print(tf.squeeze(c).eval())
    # print(tf.squeeze(a,[0]).eval())

