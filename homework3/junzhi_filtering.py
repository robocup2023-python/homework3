import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img1=cv.imread("./textjojo_new.jpg",cv.COLOR_BGR2GRAY)
img=cv.imread("./textjojo_new.jpg",cv.COLOR_BGR2GRAY)
g_nTrackbarMaxValue=9
g_nTrackbarValue=0
g_nKernelValue=0
windowname="Mean filtering"
def on_KernelTrackbar(x):
    global g_nKernelValue
    g_nTrackbarValue=cv.getTrackbarPos("res",windowname)#调用回调函数去接收制定滑动条的值
    g_nKernelValue=g_nTrackbarValue*2+1
    ksize=(g_nKernelValue,g_nKernelValue)
    cv.blur(img1,ksize,img)
cv.namedWindow("src")
cv.imshow("src",img1)
cv.namedWindow(windowname)
cv.setTrackbarPos("res",windowname,0,9,on_KernelTrackbar) #创建一个滑动条，最后一个是滑动条的回调函数指针，每当滑块位置发生变化，对应函数都会进行一次回调，而所对应的函数原型必须是void
while True:
    cv.imshow(windowname,img)
    if cv.waitKey(1)==ord("q"):
        break
cv.destroyAllWindows()