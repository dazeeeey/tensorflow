from typing import Any
import tensorflow as tf 
import numpy as np 
import pandas as pd

def relu(x):
    return tf.where(x>=0,x=0)

class MLP():
    def __init__(self,neurous = [1,100,100,1],activion=[relu,relu,None]):
       self.W=[]
       self.activion = activion
       for i in range(1,len(neurous)):
           self.W.append(tf.Variable(np.random.randn(neurous[i-1], neurous[i]))) #w
           self.W.append(tf.Variable(np.random.randn(neurous[i])))
    def __call__(self,x):
        for i in range(0,len(self.W),2):
            x = x @ self.W[i] + self.W[i+1]
            if self.activion[i // 2 ] is not None:
                x = self.activion[i // 2](x)
        return x 
    def fit(self ,X,Y,lr=0.0001,epochs=2000):
        for epoch in range(epochs):
            with tf.GradientTape() as t:
                loss = tf.reduce_mean((self(X)-Y)**2)
            dw = t.gradient(loss , self.W)
            for i , W in enumerate(self.W):
                W.assign_sub(lr * dw[i])
            if epoch% 1000 == 0:
                print(epoch , loss.numpy())




df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header=None)


itrain = []
itest = []

X = df.iloc[:,:4].values
L = df.iloc[:,-1].values

classes = np.unique(L)
spilt = 0.5
Y = []
for c in classes:
    Idx = L == c
    idx = np.where(Idx)[0]
    sp = int(spilt * len(idx))
    itrain.extend(idx[:sp])
    itest.extend(idx[sp:])
    Y.append(Idx.astype(np.int32)) #one hot     
Y = np.array(Y).T

model = MLP([4,100,50,3],[tf.sigmoid , tf.sigmoid, tf.sigmoid])
model.fit(X[itrain], Y[itrain], lr=0.1,epochs=5000)

Z = model(X[itest])
tf.argmax(Z, axis=1)

tf.argmax(Y[itest], axis=1)
np.sum(tf.argmax(Y[itest], axis=1) == tf.argmax(Z, axis=1)) / len(itest)