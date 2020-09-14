import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import interactive
image = cv2.imread("fruits.jpg")
image1 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#plt.imshow(image1, cmap=cm.gray)
size_im=image1.shape
pixel_values = image1.flatten()
# convert to float
pixel_values = np.float32(pixel_values)
k=5
C=np.zeros(k)
for i in range(k):
    cen=random.choice(pixel_values)

    C[i]=cen
print(C)
C_old = np.zeros(C.shape)

def dist(a, b):
    distance=np.zeros(k)
    for j in range(k):
        distance[j]=np.linalg.norm(a - b[j])

    return distance
clusters = np.zeros(len(pixel_values))
def error(x,y):
    d = np.zeros(k)
    for j in range(k):
        d[j] = np.linalg.norm(x[j] - y[j])
    max_d=max(abs(d))
    return max_d
dm=error(C_old,C)

while dm>np.exp(-10):
    for i in range(len(pixel_values)):
        distances = dist(pixel_values[i], C)
        cluster = np.argmin(distances)
        clusters[i]=cluster

    C_old=C
    for i in range(k):
        points = [pixel_values[j] for j in range(len(pixel_values)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    dm = error(C, C_old)

centers = np.uint8(C)
pixels=np.zeros(pixel_values.shape)
print(centers)
list1=np.linspace(0,k-1,k)
for i in range(len(clusters)):
    for j in range(len(list1)):
        if clusters[i]==list1[j]:
            pixels[i]=centers[j]

final=pixels.reshape(size_im)

plt.imshow(final.astype('uint8'))
#plt.imshow(final, cmap=cm.gray)
plt.show()
