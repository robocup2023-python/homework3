import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math
img=cv.imread("./textjojo_new.jpg",cv.IMREAD_UNCHANGED)
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
def guasskernel(size):
    sigma=1.0
    guasskernel=np.zeros((size,size),np.float32)
    for i in range(size):
        for j in range(size):
            norm=math.pow(i-size//2,2)+pow(j-size//2,2)
            guasskernel[i,j]=math.exp(-norm/(2*math.pow(sigma,2)))
    sum=np.sum(guasskernel)
    kernel=guasskernel/sum
    return kernel
def guass(img):
    h=img.shape[0]
    w=img.shape[1]
    img1=np.zeros((h,w),np.uint8)
    kernel=guasskernel(3)
    for i in range(1,h-1):
        for j in range(1,w-1):
            sum=0
            for k in range(-1,2):
                for l in range(-1,2):
                    sum+=img[i+k,j+l]*kernel[k+1,l+1]
            img1[i,j]=int(sum)
    return img1

def sobel(img_gray):
    """     sobelkernelx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobelkernely=np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) """
    img_sobelx=np.zeros((img_gray.shape[0],img_gray.shape[1]),np.float32)
    img_sobely=np.zeros((img_gray.shape[0],img_gray.shape[1]),np.float32)
    """     absX=np.zeros((img_gray.shape[0],img_gray.shape[1]),np.float32)
    absY=np.zeros((img_gray.shape[0],img_gray.shape[1]),np.float32)
    img_gray=img_gray/255

    for i in range(1,img_gray.shape[0]-1):
        for j in range(1,img_gray.shape[1]-1):
            sum=0
            for k in range(-1,2):
                for l in range(-1,2):
                    sum+=img_gray[i+k,j+l]*sobelkernelx[k+1,l+1]
            img_sobelx[i,j]=sum
    for i in range(1,img_gray.shape[0]-1):
        for j in range(1,img_gray.shape[1]-1):
            sum=0
            for k in range(-1,2):
                for l in range(-1,2):
                    sum+=img_gray[i+k,j+l]*sobelkernely[k+1,l+1]
            img_sobely[i,j]=sum """
    img_sobelx=cv.Sobel(img_gray,cv.CV_64F,1,0)
    img_sobely=cv.Sobel(img_gray,cv.CV_64F,0,1)
    absX=cv.convertScaleAbs(img_sobelx)
    absY=cv.convertScaleAbs(img_sobely)

    Sobel_img = cv.addWeighted(absX,0.5,absY,0.5,0)
    
    return Sobel_img,img_sobelx,img_sobely

def directions(x,y):#输入梯度坐标x,y,输出对于那两个点与原点的增量
    part=[[],[]]
    
    tanyx=y/(x+1e-5)
    if math.tan(-22.5)<=tanyx<=math.tan(22.5):
        part[0]=[1,0]
        part[1]=[-1,0]
    elif math.tan(22.5)<tanyx<math.tan(67.5):
        part[0]=[1,1]
        part[1]=[-1,-1]
    elif math.tan(-22.5)<tanyx<math.tan(-67.5):
        part[0]=[1,-1]
        part[1]=[-1,1]
    else:
        part[0]=[0,1]
        part[1]=[0,-1]
    return part
def reduce(Sobel_img,img_sobelx,img_sobely):
    h=Sobel_img.shape[0]
    w=Sobel_img.shape[1]
    reduce_img=np.zeros((h,w),np.uint8)
    for i in range(1,Sobel_img.shape[0]-1):
        for j in range(1,Sobel_img.shape[1]-1):
            dx=img_sobelx[i,j]
            dy=img_sobely[i,j]
            part=directions(dx,dy)
            dot1=Sobel_img[i+part[0][0],j+part[0][1]]#点1的梯度灰度值

            dot2=Sobel_img[i+part[1][0],j+part[1][1]]#点2的梯度灰度值
            if Sobel_img[i,j]>=dot1 and Sobel_img[i,j]>=dot2:
                reduce_img[i,j]=Sobel_img[i,j]
            else:
                reduce_img[i,j]=0
    return reduce_img

def mark123(reduce_img,max,min):#1/强弱标注
    h=reduce_img.shape[0]
    w=reduce_img.shape[1]
    mark_img=np.zeros((h,w),np.uint8)
    for i in range(0,reduce_img.shape[0]):
        for j in range(0,reduce_img.shape[1]):
            if reduce_img[i,j]>=max:
                mark_img[i,j]=3
            elif reduce_img[i,j]<=min:
                mark_img[i,j]=1
            else:
                mark_img[i,j]=2
    return mark_img
def changemarks(mark_img):#2强弱标注转换
    h=mark_img.shape[0]
    w=mark_img.shape[1]
    save_img123=np.zeros((h,w),np.uint8)
    surroundings=[[0,1],[0,-1],[1,0],[1,1],[1,-1],[-1,0],[-1,-1],[-1,1]]
    for i in range(1,h-1):
        for j in range(1,w-1):
            save_img123[i,j]=mark_img[i,j]
            if mark_img[i,j]==2:
                sum=0
                for l in range(len(surroundings)):
                    if mark_img[i+surroundings[l][0],j+surroundings[l][1]]==3:
                        sum=1
                        break
                if sum==1:
                    save_img123[i,j]=3
    return save_img123
def save(reduce_img,max,min):#总：nms标注
    h=reduce_img.shape[0]
    w=reduce_img.shape[1]
    mark_img=mark123(reduce_img,max,min)#通过123三种编码，标注其是弱是强
    save_img123=changemarks(mark_img)#通过遍历周围，改变标注
    save_img=np.zeros((h,w),np.uint8)
    for i in range(h):
        for j in range(w):
            if save_img123[i,j]==3:
                save_img[i,j]=reduce_img[i,j]
            else:
                save_img[i,j]=0
    return save_img

if __name__=="__main__":
    max=int(input("max: "))
    min=int(input("min: "))
    img=cv.imread("textjojo_new.jpg")
    print("yinyang has been read...")
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    print("It has been converted to gray...")

    img_guass=cv.GaussianBlur(img,(5,5),1.3)
    print("Gauss is successful!")
    cv.imwrite("guass.jpg",img_guass)
    plt.subplot(221);plt.imshow(img_guass,cmap="gray")

    """     img_guass=cv.imread("guass.jpg")
    img_guass=cv.cvtColor(img_guass,cv.COLOR_BGR2GRAY)  """ 
    Sobel_img,img_sobelx,img_sobely=sobel(img_guass)#sobel边缘检测
    print("Sobel is successful!")
    cv.imwrite("sobel.jpg",Sobel_img)    
    plt.subplot(222);plt.imshow(Sobel_img,cmap="gray")

    reduce_img=reduce(Sobel_img,img_sobelx,img_sobely)#边缘细化
    print("reduce is successful!")
    cv.imwrite("reduce.jpg",reduce_img)    
    plt.subplot(223);plt.imshow(reduce_img,cmap="gray")


    save_img=save(reduce_img,max,min)#边缘拼接
    print("save is successful!")
    cv.imwrite("save.jpg",save_img)
    plt.subplot(224);plt.imshow(save_img,cmap="gray")
    plt.show()
   









    
    
    





