#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import random


# In[88]:


def unit_step(exp):
    if exp < 0:
        return 0
    elif exp >= 0:
        return 1


# In[89]:


def error(Y, target):
    temp = []
    n = len(target)
    for i in range(0, n):
        exp = target[i] - Y[i]
        temp.append(exp)
    e = np.array(temp)
    return e


# In[90]:


def output(W, X, n):
    temp = []
    for i in range(0, n):
        exp = W[0]*X[i][0] + W[1]*X[i][1] + W[2]*X[i][2]
        temp.append(unit_step(exp))
    y = np.array(temp)
    return y


# In[91]:


def weight_init():
    temp = []
    for i in range(0,3):
        temp.append(random.uniform(-1, 1))
    
    W = np.array(temp)
    return W


# In[92]:


def weight_update(W, X, e, eta):
    temp = []
    n = len(e)
    for i in range(0, n):
        m = X[i].shape[0]
        for j in range(0, m):
            W[j] = W[j] + eta * e[i] * X[i][j]
    return W


# In[93]:


t1 = [1, 1, 1, 0]
target = np.array(t1)
n = len(target)
strTarget = "Target:\n{}\n" + 64*"-"
print(strTarget.format(target))

inputs = [[0,0], [0,1], [1,0], [1,1]]
X = np.array(inputs)
strx = "X:\n{}\n" + 64*"-"
print(strx.format(X))

W = weight_init()
strW = "Initial Weights:\n{}\n" + 64*"-"
print(strW.format(W))


# In[94]:


bias = 1.5
strBias = "Bias = {}"
print(strBias.format(bias))

temp = []
for i in range(0, len(X)):
    temp.append(np.append(X[i], bias).tolist())
X = np.array(temp)


eta = 0.2
strEta = "Eta = {}\n"
print(strEta.format(eta))

counter = 1

while True:
    Y = output(W, X, n)
    e = error(Y, target)
    
    strC = 64*"-" + "\n{}.\n" 
    print(strC.format(counter))
    
    strY = "Output Vector:\n{}\n"
    print(strY.format(Y))
    
    stre = "Error Vector:\n{}\n"
    print(stre.format(e))
    comparison = e == np.zeros(4)
    if comparison.all():
        break
    else:
        counter += 1
    W = weight_update(W, X, e, eta)
    
    strW = "Weight Vector after update:\n{}\n"
    print(strW.format(W))
    


# In[ ]:




