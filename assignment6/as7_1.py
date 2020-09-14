import random
import numpy as np
import matplotlib.pyplot as plt
import math



op=[(0, 0, 10),(0, 50, 15),(50, 0, 15),(0,-50, 15),
        (-50, 0, 15), (70,-70, 20),(-70,70, 20), (-70, -70, 20), (70,70,20),(35, 35, 15),
        (35, -35, 15),(-35, 35, 15),(-35, -35, 15)]
plt.figure()
for i in op:
    x, y, r = i[0], i[1], i[2]
    t= np.linspace(0,2*math.pi)
    plt.plot(r*np.cos(t) + x, r*np.sin(t) + y)


