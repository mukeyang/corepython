import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def d1():
    m1 = tf.constant([[3., 3.]])
    m2 = tf.constant([[2.], [2.]])
    produnct = tf.matmul(m1, m2)
    with tf.Session()as sess:
        result = sess.run([produnct])
        print(result)


# d1()
def d2():
    state=tf.Variable(0,name="counter")
    one=tf.constant(1)
    new_value=tf.add(state,one)
    update=tf.assign(state,new_value)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(state))
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))
def d3():
    input1=tf.constant(3.0)
    input2=tf.constant(2.0)
    input3=tf.constant(5.0)
    intermed=tf.add(input2,input3)
    mul=tf.multiply(input1,intermed)
    with tf.Session() as sess:
        result=sess.run([mul,intermed])
        print(result)
def d4():
    input1=tf.placeholder(tf.float32)
    input2=tf.placeholder(tf.float32)
    out=tf.multiply(input1,input2)
    with tf.Session() as sess:
        print(sess.run([out],feed_dict={input1:[7.],input2:[2.]}))
d4()