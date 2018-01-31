def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
housing = fetch_california_housing()
m, n = housing.data.shape
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]


def lr():
    reset_graph()
    # housing = fetch_california_housing()
    # m, n = housing.data.shape
    print(m, n)
    bias = np.c_[np.ones((m, 1)), housing.data]

    print(housing.data.shape)
    print(scaled_housing_data.shape)

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
    Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='Y')
    # use lr
    # XT=tf.transpose(X)
    # theta=tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),Y)
    # with tf.Session() as sess:
    #     print(theta.eval())

    n_epochs = 1000
    learning_rate = 0.01
    theta1 = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0))
    y_pred = tf.matmul(X, theta1)
    error = y_pred - Y
    mse = tf.reduce_mean(tf.square(error))
    # demo 2 use sgd
    # gradient=2/m*tf.matmul(tf.transpose(X),error)
    # 3 use autodiff
    gradient = tf.gradients(mse, [theta1])[0]
    tarinOp = tf.assign(theta1, theta1 - learning_rate * gradient)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("epoch", epoch, 'mse=', mse.eval())
            sess.run(tarinOp)
        print(theta1.eval())


# lr()
def m2():
    x = tf.Variable(3, name='x')
    y = tf.Variable(4, name='y')
    f = x * x * y + y + 2
    print(x.graph is tf.get_default_graph())
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        print(f.eval())


def myfunc(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z


def mf2():
    reset_graph()

    a = tf.Variable(0.2, name="a")
    b = tf.Variable(0.3, name="b")
    z = tf.constant(0.0, name="z0")
    for i in range(100):
        z = a * tf.cos(z + i) + z * tf.sin(b - i)

    grads = tf.gradients(z, [a, b])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        print(z.eval())
        print(sess.run(grads))


# mf2()



# print(m,n)
# m2()
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "tf"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "image", CHAPTER_ID, fig_id, +".png")
    print("save", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def mbgd():
    n_epochs = 1000
    learning_rate = 0.01
    reset_graph()
    x = tf.placeholder(tf.float32, shape=(None, n + 1),name='x')
    y = tf.placeholder(tf.float32, shape=(None, 1),name='y')
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0),name='theta')
    print(theta)
    y_pred = tf.matmul(x, theta,name='pred')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error),name='mse')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)
    init = tf.global_variables_initializer()
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))
    saver = tf.train.Saver()
    # saver = tf.train.Saver({"weights":theta})

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)
        indices = np.random.randint(m, size=batch_size)
        X_batch = scaled_housing_data_plus_bias[indices]  # not shown
        y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
        return X_batch, y_batch

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                a ,b= sess.run([training_op,mse], feed_dict={x: X_batch, y: y_batch})
            if epoch % 100 == 0:
                print("Epoch", epoch, "mes=", b)
                save_path = saver.save(sess, "./tmp/model.ckpt")
        print(theta.eval())
        save_path = saver.save(sess, "./tmp/model_final.ckpt")

def srm():
    reset_graph()
    n_epochs = 1000  # not shown in the book
    learning_rate = 0.01  # not shown

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")  # not shown
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")  # not shown
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    print(theta.name)
    y_pred = tf.matmul(X, theta, name="predictions")  # not shown
    error = y_pred - y  # not shown
    mse = tf.reduce_mean(tf.square(error), name="mse")  # not shown
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  # not shown
    training_op = optimizer.minimize(mse)  # not shown

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())  # not shown
                save_path = saver.save(sess, "./tmp/my_model.ckpt")
            sess.run(training_op)

        best_theta = theta.eval()
        save_path = saver.save(sess, "./tmp/my_model_final.ckpt")
        print(best_theta)
# srm()
# mbgd()
# reset_graph()
# saver=tf.train.import_meta_graph("./tmp/model_final.ckpt.meta")
# theta=tf.get_default_graph().get_tensor_by_name("")

reset_graph()
# notice that we start with an empty graph.

# saver = tf.train.import_meta_graph("./tmp/my_model_final.ckpt.meta")  # this loads the graph structure
# theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book
#
# with tf.Session() as sess:
#     saver.restore(sess, "./tmp/my_model_final.ckpt")  # this restores the graph's state
#     best_theta_restored = theta.eval() # not shown in the book
#     print(best_theta_restored)
def vis():
    reset_graph()

    from datetime import datetime

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./log"
    logdir = "{}/run-{}/".format(root_logdir, now)

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))
    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_index)
        indices = np.random.randint(m, size=batch_size)
        X_batch = scaled_housing_data_plus_bias[indices]  # not shown
        y_batch = housing.target.reshape(-1, 1)[indices]  # not shown
        return X_batch, y_batch
    with tf.Session() as sess:  # not shown in the book
        sess.run(init)  # not shown

        for epoch in range(n_epochs):  # not shown
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()
        print(best_theta)
        file_writer.close()
# vis()
def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)  # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")  # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")  # not shown
        return tf.maximum(z, 0., name="max")  # not shown
n_features=3
X=tf.placeholder(tf.float32,shape=(None,n_features),name='x')
with tf.variable_scope('relu'):
    threshold = tf.get_variable("threshold", shape=(),initializer=tf.constant_initializer(0.0))
with tf.variable_scope('relu',reuse=True):
    threshold = tf.get_variable("threshold", shape=(),initializer=tf.constant_initializer(0.0))
    print(threshold.name)
# with tf.variable_scope("relu", reuse=True):
#     threshold = tf.get_variable("threshold")
# with tf.variable_scope("relu") as scope:
#     scope.reuse_variables()
#     threshold = tf.get_variable("threshold")

