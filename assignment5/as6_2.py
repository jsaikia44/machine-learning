import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import random
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

mat=np.zeros((45045,150))
mat_test=np.zeros((45045,15))
import os
path='E:\pycharm\Yale'
files=[]
file_train=[]
file_test=[]
for r,d,f in os.walk(path):
    for file in f:
        if '.pgm' in file:
            files.append(os.path.join(r,file))
for i in files:
    if i.find("wink")==-1:
        file_train.append(i)
    else:
        file_test.append(i)
for i in range(0,150):
    img = cv2.imread(file_train[i])
    imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1=imggray.flatten()
    img_norm=normalize(img1)
    mat[:,i]=img_norm

for i in range(0,15):
    img_test = cv2.imread(file_test[i])
    imggray_test=cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)
    img1_test=imggray_test.flatten()
    img_norm_test=normalize(img1_test)
    mat_test[:,i]=img_norm_test
#plt.imshow(imggray,cmap=cm.gray)
#plt.show()


k=10
C=np.zeros((45045,k))
for i in range(k):
    cen=np.random.randint(0,150,1)
    C[:,i]=mat[:,int(cen)]
C_old = np.zeros(C.shape)
def error(x,y):
    dis = np.zeros(k)
    for vec in range(k):
        dis[vec] = np.linalg.norm(x[:,vec] - y[:,vec])
    max_d=max(abs(dis))
    long_d=float(format(max_d, '.8f'))
    return long_d
dm=error(C_old,C)
print(dm)
clusters=np.zeros(150)
iter=0
while dm>.0001:
    iter=iter+1
    for i in range(0,150):
        dot_pro=np.zeros(k)
        for j in range(0,k):
            dot_pro[j]=np.dot(C[:,j],mat[:,i])
        cluster = np.argmax(dot_pro)
        clusters[i]=cluster

    C_old=C
    C=np.zeros((45045,k))
    for i in range(k):
        l = 0
        points = np.zeros((45045, 150))
        for j in range(0,150):

            if clusters[j] == i:
                points[:,l] = mat[:,j]
                l = l + 1
        point_x=points[:,0:l]

        C[:,i] = np.mean(point_x, axis=1)
    C=normalize(C)
    dm = error(C, C_old)
    print("error term: ",dm)
print("total iteration: ", iter)
C_t=np.transpose(C)
mat_test_t=np.transpose(mat_test)
F=np.zeros((15,k))
for i in range(15):
    for j in range(k):
        F[i,j]=.5+.5*(np.matmul(mat_test_t[i,:],C[:,j]))

X_embedded = TSNE(n_components=2).fit_transform(F)
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig, ax = plt.subplots()
ax.scatter(X_embedded[:,0],X_embedded[:,1])

for z in range(15):
    ab = AnnotationBbox(OffsetImage(cv2.imread(file_test[z]),zoom=.2), (X_embedded[:,0][z],X_embedded[:,1][z]), frameon=True)
    ax.add_artist(ab)
plt.show()