import tensorflow as tf 
import numpy

matrix1 = tf.constant([[1.0,2.0,3.0],
                       [4.0,5.0,6.0],
                       [7.0,8.0,9.0]])

matrix2 = tf.constant([[1.0,2.0],
                       [4.0,5.0],
                       [7.0,8.0]])

reslut = tf.matmul(matrix1,matrix2)

print(matrix1.numpy())
print('x')
print(matrix2.numpy())
print("=")
print(reslut.numpy())