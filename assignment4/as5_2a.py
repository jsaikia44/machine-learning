import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
mat=np.zeros((45045,150))
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
    mat[:,i]=img1
cov_mat=np.cov(np.transpose(mat))
print(cov_mat.shape)
eigval,eigvec=np.linalg.eig(cov_mat)

idx=np.argsort(eigval)
eigval=eigval[idx]
eigvec=eigvec[:,idx]

j=149
sum_k=0
total=sum(eigval)
mat1=np.zeros((45045,14))
l=0
for k in range(1,100):

    #scale_eig = np.zeros((45045,1))
    sum_k=sum_k+eigval[j]
    scale_eig=np.matmul(mat,eigvec[j])
    mat1[:,l]=scale_eig
    l=l+1
    final=scale_eig.reshape((231,195))

    plt.imshow(final, cmap=cm.gray)
    plt.pause(.2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

    #print(final.shape)
    j = j-1
    if sum_k/total>.9:
        print(k)
        break
plt.clf()
alpha_mat=np.zeros((14,15))

for i in range(0,15):
    img1 = cv2.imread(file_test[i])
    imggray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img11=imggray1.flatten()

    mat_ps=np.linalg.pinv(mat1)
    alpha=np.matmul(mat_ps,img11)
    alpha_mat[:,i]=alpha

X_embedded = TSNE(n_components=2).fit_transform(np.transpose(alpha_mat))
print(X_embedded.shape)
plt.scatter(X_embedded[:,0],X_embedded[:,1])
plt.show()
