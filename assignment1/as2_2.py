import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-10,10,100)
mu=0
sigma=1

#e=np.zeros(shape=(200,100))

a=np.arange(start=-10, stop=10, step=.1)
b=np.arange(start=-10, stop=10, step=.1)
g=0
total=np.zeros(shape=(200,200))
for k in range(200):
    for i in range(200):
        y=np.zeros(shape=len(x))
        y_hat=np.zeros(shape=len(x))
        sum=0
        j=0
        e=0
        n = np.random.normal(loc=mu, scale=sigma, size=100)
        for j in range(100):
            y[j]=2*x[j]+3+sigma*n[j]
            y_hat[j]=a[k]*x[j]+b[i]
            e=y[j]-y_hat[j]
            err=e*e
            sum=sum+err
        av_err=sum/100
        total[i][k]=av_err
#print(total)

from mpl_toolkits.mplot3d import Axes3D
A,B = np.meshgrid(a,b)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(A,B, total)
ax.set_xlabel('a value')
ax.set_ylabel('b value')
ax.set_zlabel('error value')

plt.show()



