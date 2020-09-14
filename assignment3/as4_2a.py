from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import math as ma
import pandas as pd
iris=load_breast_cancer()
#print(iris)
x,y=iris.data,iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2, random_state=6)
print(x_train.shape)
print(y_train.shape)
mean_d1=[]
mean_d2=[]
mean_d3=[]
mean_u=[]
df = pd.DataFrame(x_train)
df1=x_train[y_train==0]
df2=x_train[y_train==1]
#df3=x_train[y_train==2]

def guassian(X_test,mu_k ,cov_k):
    mu = X_test - mu_k
    p = np.dot(mu ,np.linalg.inv(cov_k))
    value = (ma.exp(-1 *0.5 *np.dot(p ,np.transpose(mu))))/ ((2*ma.pi)**4 * np.linalg.det(cov_k))**0.5
    return value

for i in range(0,30):
    ax = df[i]
    ax1 = sum(ax) / len(df)
    mean_u.append(ax1)

    a=df1[:,i]
    a1=sum(a)/len(df1)
    mean_d1.append(a1)
    b = df2[:, i]
    b1 = sum(b) / len(df2)
    mean_d2.append(b1)
    '''c = df3[:, i]
    c1 = sum(c) / len(df3)
    mean_d3.append(c1)'''
print(mean_u)
print(mean_d1)
print(mean_d2)
#print(mean_d3)
cov_1=np.cov(np.transpose(df1))
cov_2=np.cov(np.transpose(df2))
#cov_3=np.cov(np.transpose(df3))

y_pred = np.zeros(len(x_test))
counter_test = 0
for i in range(0, len(x_test)):
    result1 = guassian(x_test[i], mean_d1, cov_1)*(len(df1)/len(x_train))
    result2 = guassian(x_test[i], mean_d2, cov_2)*(len(df2)/len(x_train))
    #result3 = guassian(x_test[i], mean_d3, cov_3)*(len(df3)/len(x_train))
    p = max([result1, result2])
    if p == result1:
        y_pred[i] = 0
    else:
        y_pred[i] = 1

print("test")
print(y_test)
print("predicted")
print(y_pred)

count = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        count = count + 1

accuracy = (count / len(y_test)) * 100

print("Accuracy")
print(accuracy)
print("confusion matrix")
print(confusion_matrix(y_test, y_pred))