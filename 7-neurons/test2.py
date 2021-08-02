import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
list1 = [2,3,13,1,16,17,25,14,18,27,26,4,6,8,11,19,24]
list2 = []
costs = []

for i in range(len(data)):
    if i not in list1:
        list2.append(i)

#list2 = list2[:5]
print(list2)
w,b = list(np.load('data2.npy'))

def dataset(file,li):
    for i in file.columns:
        data[i] = (data[i]-data[i].min())/(data[i].max()-data[i].min())
    df = pd.DataFrame([])
    for m in li:
        df = df.append(data.iloc[m-1])
    x_train = df[['Vc','f','deg','cc']]
    y_train1 = df[['Fc','Ra']]
    x_train = np.array(x_train)
    x_train = x_train.T
    y_train1 = np.array(y_train1)
    y_train1 = y_train1.T
    return x_train,y_train1

x,y = dataset(data,list2)
print(x)
print(y)

def forward_propagate(w1,b1,x):
    z = np.dot(w1,x) + b1
    output = activation(z)
    return output
def activation(z):
    return 1.0/(1+np.exp(-z))
def cost(y,y_prime,m):
    return ((y-y_prime)**2)/(2*m)


hidden_output = forward_propagate(w[0],b[0],x)
output = forward_propagate(w[1],b[1],hidden_output)
print(output)
print('printing out the cost for the test dataset')
totalCost = np.sum(cost(y,output,x.shape[1]),axis=1)
print(totalCost)
print('compare above and below')
print(y)
print(y[0,:])
plt.subplot(1,2,1)
plt.plot(range(len(y[0,:])),y[0,:],'k*-',label='target temp')
plt.plot(range(len(output[0,:])),output[0,:],'b*-',label='predicted temp')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(y[1,:])),y[1,:],'k*-',label='target vbmax')
plt.plot(range(len(output[1,:])),output[1,:],'b*-',label='predicted vbmax')
plt.legend()
plt.show()
