import ass5_1 as f
import matplotlib.pyplot as plt
r_inner1=0
r_outer1=5
c_x=0
c_y=0
n1=250
r_inner2=4
r_outer2=8
n2=350
[p1,q1]=f.genRandPointsInRing(r_inner1,r_outer1,c_x,c_y,n1)
[p2,q2]=f.genRandPointsInRing(r_inner2,r_outer2,c_x,c_y,n2)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.scatter(p1, q1, 2, "red")
plt.scatter(p2, q2, 2, "blue")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()