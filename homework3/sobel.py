import cv2 as cv
import matplotlib.pyplot as plt
img=cv.imread("/home/robocup/桌面/vs_codekd/textjojo_new.jpg",cv.COLOR_BGR2GRAY)

rgb_img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
grayImage=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
x=cv.Sobel(grayImage,cv.CV_16S,1,0)
y=cv.Sobel(grayImage,cv.CV_16S,0,1)
absX=cv.convertScaleAbs(x)
absY=cv.convertScaleAbs(y)
Sobel=cv.addWeighted(absX,1,absY,1,0)
cv.imwrite("./sobel_pro.jpg",Sobel)
plt.rcParams['font.sans-serif']=["SimHei"]
titles=["原始图像","SOBEL算子"]
images=[grayImage,Sobel]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()