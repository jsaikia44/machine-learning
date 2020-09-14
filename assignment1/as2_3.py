import numpy as np
a=2
b=3
x=np.linspace(-10,10,100)
mu=0
sigma=1
n = np.random.normal(loc=mu,scale=sigma,size=100)
y=a*x+b+sigma *n

v=np.zeros(shape=(100,2))
for i in range(100):
    v[i][0]=1
    v[i][1]=x[i]
v_t=np.transpose(v)
v1=np.linalg.inv(np.matmul(v_t,v))
v_f=np.matmul(v1,v_t)
w=np.matmul(v_f,y)
print('a=',w[1],'\nb=',w[0])

