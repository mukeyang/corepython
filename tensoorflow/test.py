import  tensorflow as tf
hello=tf.constant("hello tensor")
sess=tf.Session()
print(sess.run(hello))
a=tf.constant(10)
b=tf.constant(21)
print(sess.run(a+b))