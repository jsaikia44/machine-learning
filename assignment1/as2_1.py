import numpy as np
import matplotlib.pyplot as plt
a=2
b=3
x=np.linspace(-10,10,100)
y=a*x+b
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.plot(x, y,color='black')
mu=0
sigma=1
n = np.random.normal(loc=mu,scale=sigma,size=100)
y1=a*x+b+sigma *n
plt.scatter(x, y1,5,color='red')
plt.show()
