import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
#best result at ::
#list1 = [1,2,3,4,22,8,11,16,17,19,24,5,14,9,10,12,15,25,26]
#list1 = [1,2,3,4,22,8,17,16,26,25,24,5,14,9,10,12,15,13,7]
list1 = [1,2,3,4,5,6,8,9,11,12,13,14,16,17,18,19,20,22,24,25,26,27]
list2 = []

def dataset(file,li):
    for i in file.columns:
        file[i] = (file[i]-file[i].min())/(file[i].max()-file[i].min())
    file[file==0] = 1e-10
    df = pd.DataFrame([])
    for m in li:
        df = df.append(file.iloc[m-1])
    x_train = df[['Vc','f','deg','cc']]
    y_train1 = df[['Ra','Vbmax']]
    x_train = np.array(x_train)
    x_train = x_train.T
    y_train1 = np.array(y_train1)
    y_train1 = y_train1.T
    return x_train,y_train1
np.random.seed(10)

n_input = 4
n_hidden = 15
n_output = 2
w,b = [],[]
costs = []
n_epochs = 15000
lr = 0.1

w1 = np.random.rand(n_hidden,n_input)
w.append(w1)
w2 = np.random.rand(n_output,n_hidden)
w.append(w2)
b1 = np.random.rand(n_hidden,1)
b.append(b1)
'''b2 = np.random.rand(n_output,1)
b.append(b2)'''

x,y = dataset(data,list1)

def activation(z):
    return 1.0/(1.0+np.exp(-z))

def cost(y,y_prime,m):
    return ((y-y_prime)**2)/(2*m)

def cost_derivative_y_prime(y,y_prime,m):
    return -(y-y_prime)/(2*m)

def activation_derivative(z):
    return z*(1-z)

def forward_propagate(weight,x,bias=None):
    if bias is not None:
        z = np.dot(weight,x) + bias
        output = activation(z)
    else:
        z = np.dot(weight,x)
        output = z
    return output

def backward_propagate(y,y_prime,input,m,weights=None):

    if weights is not None:
        dcostdw = np.dot((np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*1))*input*(1-input)),x.T)
        #dcostdw = np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*activation_derivative(y_prime)))*input*(1-input)
        dcostdb = np.sum(np.dot(weights.T,(cost_derivative_y_prime(y,y_prime,m)*1))*input*(1-input),axis=1)
        return dcostdw,dcostdb.reshape(1,n_hidden).T
    else:
        dcostdw =  np.dot(cost_derivative_y_prime(y,y_prime,m)*1,input.T)
        dcostdb =  np.sum((cost_derivative_y_prime(y,y_prime,m)*1),axis=1)
        return dcostdw,dcostdb.reshape(2,1)

for i in range(n_epochs):
    hidden_output = forward_propagate(w[0],x,b[0])
    output = forward_propagate(w[1],hidden_output)
    totalCost = np.sum(cost(y,output,x.shape[1]))
    costs.append(totalCost)
    wei,bia = backward_propagate(y,output,hidden_output,x.shape[1])
    hidden_weight,hidden_bias = backward_propagate(y,output,hidden_output,x.shape[1],weights=w[1])
    w[1] = w[1] - (lr*wei)
    #b[1] = b[1] - (lr*bia)
    w[0] = w[0] - (lr*hidden_weight)
    b[0] = b[0] - (lr*hidden_bias)

hidden_output = forward_propagate(w[0],x,b[0])
output = forward_propagate(w[1],hidden_output)

print(costs[-2:])
plt.plot(range(n_epochs),costs)
plt.show()

plt.subplot(1,2,1)
plt.plot(range(len(y[0,:])),y[0,:],'k*-',label='target Fc')
plt.plot(range(len(output[0,:])),output[0,:],'b*-',label='predicted Fc')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(y[1,:])),y[1,:],'k*-',label='target Ra')
plt.plot(range(len(output[1,:])),output[1,:],'b*-',label='predicted Ra')
plt.legend()
plt.show()

from numpy import asarray
from numpy import save

data1 = []
data1.append(w)
data1.append(b)

data = asarray(np.array(data1))
save('data1.npy',data)

data = pd.read_csv('data.csv')

for i in range(1,len(data)+1):
    if i not in list1:
        list2.append(i)
print(list2)
x,y = dataset(data,list2)
hidden_output = forward_propagate(w[0],x,b[0])
output = forward_propagate(w[1],hidden_output)

plt.subplot(1,2,1)
plt.plot(range(len(y[0,:])),y[0,:],'k*-',label='target temp')
plt.plot(range(len(output[0,:])),output[0,:],'b*-',label='predicted temp')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(y[1,:])),y[1,:],'k*-',label='target vbmax')
plt.plot(range(len(output[1,:])),output[1,:],'b*-',label='predicted vbmax')
plt.legend()
plt.show()

print(output)
print('compare above and below')
print('compare above and below')
print('compare above and below')
print('compare above and below')
print('compare above and below')
print(y)
