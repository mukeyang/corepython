import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
a=tf.constant([[1,1,1],[2,2,2],[3,3,3]])
with tf.Session()as sess:
    print(tf.reduce_sum(a,1).eval())
    print(tf.reduce_sum(a,0).eval())