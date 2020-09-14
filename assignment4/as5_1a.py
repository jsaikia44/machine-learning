import random
import numpy as np
import matplotlib.pyplot as plt
from math import pi
a=[]
b=[]
a1=40     #radius on the x-axis
b1=20    #radius on the y-axis
t = np.linspace(0, 2*pi, 100)
list1=np.linspace(100,1000,10)
for n in list1:
    n=int(n)
    mat = np.zeros(shape=(n, 2))
    while len(a)<n:
        x1 = round(random.uniform(60, 140), 2)
        y1 = round(random.uniform(60, 140), 2)
        if ((((x1-100)*.707+(y1-100)*.707)/a1)**2 + (((x1-100)*.707-(y1-100)*.707)/b1)**2) < 1:
            a.append(x1)
            b.append(y1)
    mat[:,0]=a
    mat[:,1]=b
    cov_mat=np.cov(np.transpose(mat))
    print(cov_mat)
    plt.scatter(a,b,5)
    plt.title('Scatter plot ')
    plt.xlabel('x')
    plt.ylabel('y')

    lamda,v=np.linalg.eig(cov_mat)
    e_max=np.sqrt(max(lamda))
    x=.2*e_max
    origin = [100], [100] # origin point
    plt.quiver(*origin, v[:,0], v[:,1], color=['r','b'],scale=x)
    plt.pause(.5)
    plt.clf()

plt.show()
