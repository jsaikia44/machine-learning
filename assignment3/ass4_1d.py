import numpy as np
import ass5_1 as f
import matplotlib.pyplot as plt
k=np.arange(.01,10,.05)
r_inner1=0
r_outer1=4
c_x=0
c_y=0
n1=250
r_inner2=5
r_outer2=8
n2=350
[p1,q1]=f.genRandPointsInRing(r_inner1,r_outer1,c_x,c_y,n1)
mean_p1=sum(p1)/250
mean_q1=sum(q1)/250
[p2,q2]=f.genRandPointsInRing(r_inner2,r_outer2,c_x,c_y,n2)
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.scatter(p1, q1, 2, "red")
plt.scatter(p2, q2, 2, "blue")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
#interactive(False)
fig.show()
mean_p2=sum(p2)/350
mean_q2=sum(q2)/350
tp_list=np.zeros(200)
fp_list=np.zeros(200)
for i in range(200):
    tp=0
    fp=0
    for j in range(250):
        if (p1[j]-mean_p1)<=k[i] and (p1[j]-mean_p1)>=-k[i] and (q1[j]-mean_q1)<=k[i] and (q1[j]-mean_q1)>=-k[i]:
            tp=tp+1
    for l in range(350):
        if (p2[l]-mean_p2)<=k[i] and (p2[l]-mean_p2)>=-k[i] and (q2[l]-mean_p2)<=k[i] and (q2[l]-mean_p2)>=-k[i]:
            fp=fp+1
    tp_rate=tp/250
    fp_rate=fp/350
    tp_list[i]=tp_rate
    fp_list[i]=fp_rate
fig2 = plt.figure(2)
plt.plot(fp_list,tp_list)
plt.show()




