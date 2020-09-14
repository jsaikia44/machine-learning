import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as interactive
b_lo =-10
b_up = 10

def f(x,y):
    return -20*math.exp(-0.2*math.sqrt(0.5*(x*x+y*y))) - math.exp(0.5*(math.cos(2*math.pi*x)+math.cos(2*math.pi*y))) + math.exp(1)+20

x_g=np.random.uniform(b_lo,b_up)
y_g=np.random.uniform(b_lo,b_up)

S = 50
ini_pos_x=np.zeros(S)
ini_pos_y=np.zeros(S)
p_x=np.zeros(S)
p_y=np.zeros(S)
f_p = np.zeros(S)
v_i_x = np.zeros(S)
v_i_y = np.zeros(S)

for i in range(1,S):
    ini_pos_x[i]=np.random.uniform(b_lo,b_up)
    ini_pos_y[i]=np.random.uniform(b_lo,b_up)    
    p_x[i]=ini_pos_x[i]
    p_y[i]=ini_pos_y[i]
    if f(ini_pos_x[i],ini_pos_y[i])<f(x_g,y_g):
        x_g = ini_pos_x[i]
        y_g = ini_pos_y[i]
    v_i_x[i] = np.random.uniform(-abs(b_up-b_lo),abs(b_up-b_lo))
    v_i_y[i] = np.random.uniform(-abs(b_up-b_lo),abs(b_up-b_lo))
itr = 0 
maxItr = 100

w = 0.05
theta_p = 0.5
theta_g = 1
L = 0.05

Z= np.zeros(shape=(100,100))
a = np.linspace(-10,10,100)
b = a
for i in range(0,100):
    for j in range(0,100):
        Z[i][j] = f(a[i],b[j])

flag = 0  
while itr<maxItr:
    for i in range(1,50):
        if flag%5 == 0:
            print("pso")
            r_p = np.random.uniform(0,1)
            r_g = np.random.uniform(0,1)
            v_i_x[i] = w * v_i_x[i] + theta_p * r_p * (p_x[i]-ini_pos_x[i]) + theta_g * r_g *(x_g-ini_pos_x[i]) 
            v_i_y[i] = w * v_i_y[i] + theta_p * r_p * (p_x[i]-ini_pos_y[i]) + theta_g * r_g *(y_g-ini_pos_y[i])
            #update 
            ini_pos_x[i] = ini_pos_x[i] + v_i_x[i]
            ini_pos_y[i] = ini_pos_y[i] + v_i_y[i]

            if f(ini_pos_x[i],ini_pos_y[i])<f(p_x[i],p_y[i]):
                p_x[i]=ini_pos_x[i]
                p_y[i]=ini_pos_y[i]
                if f(p_x[i],p_y[i])<f(x_g,y_g):
                    x_g=p_x[i]
                    y_g=p_y[i]
            plt.scatter(p_x,p_y)
            X,Y = np.meshgrid(a,b)
            plt.contour(X,Y,Z)      
        elif flag%5 != 0:
            print("gd")
            ini_pos_x[i]=ini_pos_x[i] - L * math.pi * math.exp((math.cos(2 * math.pi * ini_pos_x[i])+ math.cos(2 * math.pi * ini_pos_y[i]))/2)*math.sin(2*math.pi*ini_pos_x[i]) + (math.pow(2,3/2) * ini_pos_x[i] * math.exp((-1)*math.sqrt(ini_pos_x[i]*ini_pos_x[i]+ini_pos_y[i]*ini_pos_y[i])/5*math.sqrt(2))/math.sqrt(ini_pos_x[i]*ini_pos_x[i]+ini_pos_y[i]*ini_pos_y[i]))
            ini_pos_y[i]=ini_pos_y[i] - L * math.pi * math.exp((math.cos(2 * math.pi * ini_pos_x[i])+ math.cos(2 * math.pi * ini_pos_y[i]))/2)*math.sin(2*math.pi*ini_pos_y[i]) + (math.pow(2,3/2) * ini_pos_y[i] * math.exp((-1)*math.sqrt(ini_pos_x[i]*ini_pos_x[i]+ini_pos_y[i]*ini_pos_y[i])/5*math.sqrt(2))/math.sqrt(ini_pos_x[i]*ini_pos_x[i]+ini_pos_y[i]*ini_pos_y[i]))

            if f(ini_pos_x[i],ini_pos_y[i])<f(p_x[i],p_y[i]):
                p_x[i]=ini_pos_x[i]
                p_y[i]=ini_pos_y[i]
                if f(p_x[i],p_y[i])<f(x_g,y_g):
                    x_g=p_x[i]
                    y_g=p_y[i]
            plt.scatter(p_x,p_y)
            X,Y = np.meshgrid(a,b)
            plt.contour(X,Y,Z)
        flag=flag+1
    itr = itr + 1
    plt.pause(.025)
    if itr!=maxItr-1 :
        plt.clf()
plt.show()

