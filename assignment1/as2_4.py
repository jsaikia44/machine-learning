import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-10,10,100)
n=100
a=2
b=3
a1=10
b1=10
L=.05
mu=0
sigma=1
a_list=np.zeros(100)
b_list=np.zeros(100)
total=np.zeros(shape=(100,100))
p = np.random.normal(loc=mu,scale=sigma,size=100)
y=a*x+b+sigma *p
for i in range(len(x)):
    y_hat=a1*x+b1
    d_a1=(-1/n)*sum(x*(y-y_hat))
    d_b1=(-1/n)*sum(y-y_hat)
    a1=a1-L*d_a1
    a_list[i]=a1
    b1=b1-L*d_b1
    b_list[i]=b1
print(a1,b1)
a_m=np.linspace(-10,10,100)
b_m=np.linspace(-10,10,100)
for k in range(100):
    for i in range(100):
        #y1=np.zeros(shape=len(x))
        #y_hat1=np.zeros(shape=len(x))
        sum=0
        j=0
        e=0
        n = np.random.normal(loc=mu, scale=sigma, size=100)
        for j in range(100):
            y1=2*x[j]+3+sigma*n[j]
            y_hat1=a_m[k]*x[j]+b_m[i]
            e=y1-y_hat1
            err=e*e
            sum=sum+err
        av_err=sum/100
        total[k][i]=av_err

X,Y = np.meshgrid(a_m,b_m)
fig2, ax1 = plt.subplots()
fig2.suptitle('contour plot')
ax1.contour(X,Y,total,levels=30)
#ax1.scatter(a_list,b_list,2,color='red')
plt.plot(a_list,b_list,'-',2,color='red')
plt.show()