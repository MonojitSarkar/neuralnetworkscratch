import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
list1 = [2,3,13,1,16,17,25,14,18,27,26,4,6,8,11,19,24]

def dataset(file,li):
    for i in file.columns:
        data[i] = (data[i]-data[i].min())/(data[i].max()-data[i].min())
    df = pd.DataFrame([])
    for m in li:
        df = df.append(data.iloc[m-1])
    x_train = df[['Vc','f','deg','cc']]
    y_train1 = df[['Temp','Vbmax']]
    x_train = np.array(x_train)
    x_train = x_train.T
    y_train1 = np.array(y_train1)
    y_train1 = y_train1.T
    return x_train,y_train1
np.random.seed(9)

n_input = 4
n_hidden = 9
n_output = 2
w,b = [],[]
costs = []
n_epochs = 50000
lr = 50

w1 = np.random.rand(n_hidden,n_input)
w.append(w1)
w2 = np.random.rand(n_output,n_hidden)
w.append(w2)
b1 = np.random.rand(n_hidden,1)
b.append(b1)
b2 = np.random.rand(n_output,1)
b.append(b2)

x,y = dataset(data,list1)

print('weight and bias before')
print(x)
print(y)
print(w)
print(b)

def activation(z):
    return 1.0/(1.0+np.exp(-z))

def cost(y,y_prime,m):
    return ((y-y_prime)**2)/(2*m)

def cost_derivative_y_prime(y,y_prime,m):
    return -(y-y_prime)/(2*m)

def activation_derivative(z):
    return z*(1-z)

def forward_propagate(weight,bias,input):
    z = np.dot(weight,input) + bias
    output = activation(z)
    return output

def backward_propagate(y,y_prime,input,m,weights=None):

    if weights is not None:
        dcostdw = np.dot((np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)))*input*(1-input)),x.T)
        #dcostdw = np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)))*input*(1-input)
        dcostdb = np.sum(np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)))*input*(1-input),axis=1)
        return dcostdw,dcostdb.reshape(1,n_hidden).T
    else:
        dcostdw =  np.dot(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime),input.T)
        dcostdb =  np.sum((cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)),axis=1)
        return dcostdw,dcostdb.reshape(2,1)

for i in range(n_epochs):
    hidden_output = forward_propagate(w[0],b[0],x)
    output = forward_propagate(w[1],b[1],hidden_output)
    totalCost = np.sum(cost(y,output,x.shape[1]))
    costs.append(totalCost)
    wei,bia = backward_propagate(y,output,hidden_output,x.shape[1])
    hidden_weight,hidden_bias = backward_propagate(y,output,hidden_output,x.shape[1],weights=w[1])
    w[1] = w[1] - (lr*wei)
    b[1] = b[1] - (lr*bia)
    w[0] = w[0] - (lr*hidden_weight)
    b[0] = b[0] - (lr*hidden_bias)

print('weight and bias after')
print(w)
print(b)

hidden_output = forward_propagate(w[0],b[0],x)
output = forward_propagate(w[1],b[1],hidden_output)
print(output)
print('compare above and below')
print(y)

print(costs[-2:])
plt.plot(range(n_epochs),costs)
plt.show()

plt.subplot(1,2,1)
plt.plot(range(len(y[0,:])),y[0,:],'k*-',label='target temp')
plt.plot(range(len(output[0,:])),output[0,:],'b*-',label='predicted temp')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(y[1,:])),y[1,:],'k*-',label='target temp')
plt.plot(range(len(output[1,:])),output[1,:],'b*-',label='predicted vbmax')
plt.legend()
plt.show()

from numpy import asarray
from numpy import save

data1 = []
data1.append(w)
data1.append(b)

data = asarray(np.array(data1))
save('data2.npy',data)
