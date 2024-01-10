import tensorflow as tf 

# a = tf.add(3,5)

with tf.compat.v1.Session() as sess:
    x = tf.constant(2)
    y = tf.constant(3)
    op1 = tf.multiply(x,y)
    op2 = tf.add(x,y)
    op3 = pow(op1,op2)

print(sess.)