import cv2
import numpy as np

image = cv2.imread("fruits.jpg")
row,col,size=image.shape
print(image.shape)
k=3
C=np.zeros((k,size))
C_old=np.zeros((k,size))
for i in range(k):
    pixel= image[np.random.randint(0,row),np.random.randint(0,col)]
    C[i,:]=np.array([pixel[0],pixel[1],pixel[2]])
def dist(a, b):
    distance= np.linalg.norm(a - b)
    return distance
clusters = np.zeros((row,col))
def error(x,y,ke):
    d = np.zeros(ke)
    for c in range(0,ke):
        d[c] = np.linalg.norm( x[c,:]- y[c,:])
    max_d=max(abs(d))
    return max_d
dm=error(C_old,C,k)

while dm>np.exp(-2):
    for i in range(row):
        for j in range(col):
            distances=np.zeros(k)
            for pix in range(0,k):
                distances[pix] = dist(image[i,j], C[pix,:])
            cluster = np.argmin(distances)

            clusters[i,j]=cluster

    C_old=C
    C = np.zeros((k, size))
    for i in range(0,k):
        m=0
        points=np.array([])
        for row_val in range(0,row):
            for col_val in range(0,col):
                if clusters[row_val,col_val] == i:
                    x_val = image[row_val,col_val]
                    points = np.append(points, x_val)
        length=len(points)
        points=np.reshape(points,(-1,3))

        C[i,:] = np.mean(points, axis=0)
        #C[i,:] =np.array([sum(points[:,0])/m,sum(points[:,1])/m,sum(points[:,2])/m])
    print(C_old,"old")
    print(C,"new")
    dm = error(C, C_old,k)
centers = np.uint8(C)
pixels=np.zeros(image.shape)
print(centers)
list1=np.linspace(0,k-1,k)
for i in range(row):
    for k in range(col):
        for ind in range(len(list1)):
            if clusters[i,k]==list1[ind]:
                pixels[i,k]=centers[ind,:]

pixels=np.uint8(pixels)
'''plt.imshow(pixels)
plt.show()'''
cv2.imshow('seg img',pixels)
cv2.waitKey(0)
cv2.destroyAllWindows()
