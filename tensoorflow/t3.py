import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import  tensorflow as tf
import numpy as np
import xlrd
import matplotlib.pyplot as plt
# a=numpy.array([[1,2,3],[2,2,3],[3,3,4]])
# print(a[a>2])
# print(np.arange(4).reshape(4,1))
def t1():
    a = tf.constant(2)
    b = tf.constant(3)
    x = tf.add(a, b)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        print(sess.run(x))
    writer.close()


def t3():
    print(tf.ones([2, 3]))
    w = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1))
    t = w.assign([[1, 2, 3], [1, 2, 3]])
    with tf.Session() as sess:
        sess.run(w.initializer)
        print(w.eval())
        sess.run(t)
        print(sess.run(w))


def t2():
    global a
    a = tf.placeholder(tf.float32, shape=[3])
    b = tf.constant([5, 5, 5], tf.float32)
    c = a + b
    with tf.Session()as  sess:
        print(sess.run(c, feed_dict={a: [1, 2, 3]}))
file= 'data/fire_theft.xls'
book=xlrd.open_workbook(file, encoding_override='utf-8')
sheet=book.sheet_by_index(0)
data=np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)],dtype=np.float32)
n_samples=sheet.nrows-1
# print(data)

# t1()
# t3()
# t2()
X=tf.placeholder(tf.float32,shape=[],name='input')
Y=tf.placeholder(tf.float32,shape=[],name='label')
w=tf.get_variable('weight',shape=[],initializer=tf.truncated_normal_initializer())
b=tf.get_variable('bias',shape=[],initializer=tf.zeros_initializer())
y_pred=w*X+b
loss=tf.square(Y-y_pred,name='loss')
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)
# init=tf.global_variables_initializer()
init=tf.variables_initializer([w,b])
with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter("./linear",graph=sess.graph)
    for i in  range(100):
        total_loss=0
        for x,y in data:
            _,l=sess.run([optimizer,loss],feed_dict={X:x,Y:y})
            total_loss+=l
        print("Epoch{0}:{1}".format(i,total_loss/n_samples))
    writer.close()
    w,b=sess.run([w,b])
X,Y=data.T[0],data.T[1]
plt.plot(X,Y,'bo',label='reaL DATA')
plt.plot(X,X*w+b,'r',label='predicted')
plt.legend()
plt.show()

