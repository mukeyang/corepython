import tensorflow as  tf
learn_rate=0.01
batch_size=16
epoch_step=1000
X=tf.placeholder(tf.float32,[batch_size,784],name='X')
Y=tf.placeholder(tf.int32,[batch_size,10],name='Y')
layer1=16
layer2=32
w={
    'h1':tf.Variable(tf.random_normal([784,layer1])),
    'h2':tf.Variable(tf.random_normal([layer1,layer2])),
    'out':tf.Variable(tf.random_normal([layer2,10]))
}
b={
    'h1':tf.Variable(tf.random_normal([layer1])),
    'h2':tf.Variable(tf.random_normal([layer2])),
    'out':tf.Variable(tf.random_normal([10]))
}
def network(x_input, weight, biases):
    net1=tf.nn.relu(tf.matmul(x_input, weight['h1']) + biases['h1'])
    net2=tf.nn.relu(tf.matmul(net1, weight['h2']) + biases['h2'])
    output= tf.matmul(net2, weight['out']) + biases['out']
    return output
pred=network(X,w,b)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))
optimizer=tf.train.AdamOptimizer(learn_rate).minimize(loss)
correct_preds=tf.equal(tf.argmax(Y,1),tf.argmax(pred,1))
accuracy=tf.reduce_mean((tf.case((correct_preds,tf.float32))))

