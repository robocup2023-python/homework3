import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
def My_corner_Harris(image,blocksize,ksize,k):
    gray_img=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    src=gray_img.astype(np.float32)

    SrcHeight=src.shape[0]
    SrcWidth=src.shape[1]

    Ix=cv.Sobel(src,-1,1,0,ksize)
    Iy=cv.Sobel(src,-1,0,1,ksize)
    Ix2=np.multiply(Ix,Ix)
    Ixy=np.multiply(Ix,Iy)
    Iy2=np.multiply(Iy,Iy)

    Ix2_guass=cv.GaussianBlur(Ix2,(blocksize,blocksize),1.3)
    Ixy_guass=cv.GaussianBlur(Ixy,(blocksize,blocksize),1.3)
    Iy2_guass=cv.GaussianBlur(Iy2,(blocksize,blocksize),1.3)

    R=np.zeros((SrcHeight,SrcWidth),np.float32)
    for i in range(0,SrcHeight):
        for j in range(0,SrcWidth):
            M=np.array([[Ix2_guass[i,j],Ixy_guass[i,j]],[Ixy[i,j],Iy2_guass[i,j]]])
            R[i,j]=np.linalg.det(M)-k*((M.trace()**2))
    return R
block_size=3
sobel_size=5
k=0.04

img=cv.imread("yinyang.jpg")
R=My_corner_Harris(img,block_size,sobel_size,k)
img[R>0.2*R.max()]=[0,0,225]
cv.imwrite("Harris_detect.jpg",img)
cv.imshow("Harris detect with read point",img)
cv.waitKey(0)
cv.destroyAllWindows()





    

