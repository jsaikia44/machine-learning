import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import interactive
from mpl_toolkits.mplot3d import Axes3D
x=np.linspace(-10,10,100)
y=np.linspace(-10,10,100)
z=np.zeros((100,100))
for i in range(100):
    for j in range(100):
        z[i][j]=1.7*np.exp(-((x[i]-3)**2+(y[j]-3)**2)/10)+np.exp(-((x[i]+5)**2+(y[j]+5)**2)/8)+2*np.exp(-(((x[i]**2)/4)+((y[j]**2)/5)))+1.5*np.exp(-((((x[i]-4)**2)/18)+(((y[j]+4)**2)/16)))+1.2*np.exp(-((((x[i]+4)**2)/18)+(((y[j]-4)**2)/16)))
X,Y = np.meshgrid(x,y)
fig1 = plt.figure(1)

ax = plt.axes( projection='3d')

ax.plot_surface(X,Y,z)
ax.set_xlabel('x value')
ax.set_ylabel('y value')
ax.set_zlabel('z value')
fig1.suptitle('3d surface plot')
interactive(True)
plt.show()

fig2, ax1 = plt.subplots()
fig2.suptitle('contour plot')
ax1.contour(X,Y,z)

interactive(False)
plt.show()