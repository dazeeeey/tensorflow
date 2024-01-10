import tensorflow as tf 
import numpy as np 
from matplotlib import pyplot as plt 

N = 100 
x = np.random.rand(N)
Y = 5 * x + 10 + 0.4* np.random.rand(N)

W = np.random.rand()
b = np.random.rand()

W = tf.Variable(W)
b = tf.Variable(b)
Ir = tf.constant(0.1)

for epoch in range(2000):
    with tf.GradientTape() as t:
        y = tf.add(tf.multiply(W,x),b)
        loss = tf.reduce_mean(tf.pow((y-Y),2))
    dW , db = t.gradient(loss,[W,b])
    W.assign_sub(tf.multiply(Ir,dW))
    b.assign_sub(tf.multiply(Ir,db))
    if epoch % 10 == 0 :
        print(epoch,W.numpy(),b.numpy(),loss.numpy())

Z = W*x+b
plt.plot(x,Y,'.')
plt.plot(x,Z,'.r')
plt.show()