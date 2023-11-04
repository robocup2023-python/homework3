import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img=cv.imread("./yinyang.jpg")
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges=cv.Canny(img,50,100)
plt.subplot(121)
plt.imshow(img,cmap="gray")
plt.title("Origin"),plt.xticks([]),plt.yticks([])
plt.subplot(122)
plt.imshow(edges,cmap='gray')
plt.title("NOw"),plt.xticks([]),plt.yticks([])
plt.show()
cv.imwrite("done1.jpg",edges)
cv.waitKey(0)
cv.destroyAllWindows()