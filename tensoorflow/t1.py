import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

def m1():
    a = tf.add(3, 5)
    print(a)
    b = tf.add(3.0, 5.)
    print(b)
    c = tf.add(3.0, 5.)
    print(c)
    x = 2
    y = 3
    op1 = tf.add(x, y)
    op2 = tf.multiply(x, y)
    useless = tf.multiply(x, op1)
    op3 = tf.pow(op1, op2)
    g = tf.get_default_graph()
    with tf.Session()as sess:
        print(sess.run([op3, useless]))


def m2():
    a = tf.constant([2, 3], name='a')
    b = tf.constant([[0, 1], [2, 3]], name='b')
    x = tf.add(b, a)
    c = tf.zeros([2, 3], tf.int32)
    d = tf.zeros_like(b)
    e = tf.fill([2, 3], 9)
    g = tf.linspace(10.0, 13.0, 4)
    h = tf.range(0, 8, 2)
    with tf.Session() as sess:
        print(sess.run([x, c, d, g, h]))


def m3():
    value = tf.constant([[1, 1], [2, 2], [3, 3]])
    a = tf.random_shuffle(value)
    b = tf.random_crop(value, (3, 1))
    c = tf.constant(np.random.normal(size=(3, 4)))
    d = tf.multinomial(c, 10)
    e = tf.random_gamma([10], [5, 10])
    with tf.Session()as sess:
        print(sess.run(e))


def m4():
    const = tf.constant([1.0, 2.0], name='const')
    w = tf.Variable(10)
    w1 = w.assign(2 * w.initialized_value())
    w2 = w.assign_add(10)
    init = tf.global_variables_initializer()
    # init_const=tf.variables_initializer([const],name='init')
    with tf.Session() as sess:
        # sess.run(init)
        g = sess.graph
        # sess.run(w1)
        tf.get_default_graph()
        with g.control_dependencies([w1]):
            w3 = w.assign_add(10)
            sess.run(w3)
        sess.run(w1)
        print(w.eval())


def m5():
    a = tf.placeholder(tf.float32, shape=[])
    b = tf.constant([5., 5., 5.])
    c = a + b
    with tf.Session()as sess:
        print(sess.graph.is_feedable(b))
        print(sess.run(c, {a: 2, b: [1, 2, 3]}))


# m1()
# m2()
# m3()
# m4()
m5()
