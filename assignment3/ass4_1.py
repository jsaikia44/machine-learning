def genRandPointsInRing(r_inner,r_outer,c_x,c_y,n):
    import numpy as np
    if r_inner>r_outer:
        print("invalid input")
        exit(1)
    px_new=[]
    py_new=[]
    a=r_outer+max(c_x,c_y)
    while len(px_new)<n:
        px = np.random.uniform(-a,a)
        py = np.random.uniform(-a,a)
        if (px-c_x)**2+(py-c_y)**2<=r_outer**2 and (px-c_x)**2+(py-c_y)**2>=r_inner**2:
            px_new.append(px)
            py_new.append(py)
    return [px_new,py_new]

