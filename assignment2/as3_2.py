import numpy as np
import math as ma
import matplotlib.pyplot as plt
from matplotlib import interactive
b_low=-10
b_up=10
w=.05
phi_p=.5
phi_g=1
s=100


v=np.zeros((100,2))

def f(x,y):
    return -20*np.exp(-0.2*(.5 * (x ** 2 + y ** 2))**.5)- np.exp(.5*((ma.cos( 2 * ma.pi * x)) + ma.cos( 2 * ma.pi* y))) + np.exp(1) + 20

gx_best=np.random.uniform(low=-20, high=20)
gy_best=np.random.uniform(low=-20, high=20)

x=np.zeros(s)
y=np.zeros(s)
px=np.zeros(s)
py=np.zeros(s)
A= np.linspace(-10,10,100)
B=np.linspace(-10,10,100)
c=np.zeros((100,100))
for i in range(100):
    for j in range(100):
        c[i][j]=f(A[i],B[j])
for i in range(1,s):
    x[i] = np.random.uniform(low=b_low, high=b_up)
    y[i] = np.random.uniform(low=b_low, high=b_up)
    px[i]=x[i]
    py[i]=y[i]
    if f(px[i],py[i])< f(gx_best,gy_best):
        gx_best=px[i]
        gy_best=py[i]
    v[i][0] = np.random.uniform(low=-20, high=20)
    v[i][1] = np.random.uniform(low=-20, high=20)
itr=0
maxitr=50
v_new=np.zeros(s)
while itr<maxitr:
    for j in range(s):
        rp= np.random.uniform(0,1)
        rg= np.random.uniform(0,1)
        v[j][0]=w*v[j][0]+phi_g*rg*(px[j]-x[j])+phi_g*rg*(gx_best-x[j])
        v[j][1]=w*v[j][1]+phi_g*rg*(py[j]-y[j])+phi_g*rg*(gy_best-y[j])

        x[j]=x[j]+v[j][0]
        y[j]=y[j] + v[j][1]
        if f(x[j],y[j])<f(px[j],py[j]):
            px[j]=x[j]
            py[j]=y[j]
            if f(px[j],py[j])<f(gx_best,gy_best):
                gx_best=px[j]
                gy_best=py[j]
        plt.scatter(px,py)
        a, b = np.meshgrid(A, B)
        plt.contour(a, b, c)

    itr=itr+1
    plt.pause(.08)
    if itr!=maxitr-1:
        plt.clf()
plt.show()







