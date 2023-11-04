import numpy as np
import cv2 as cv
import random

import matplotlib.pyplot as plt
def make_jiaoyanzaoyin(img,por):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if random.random()<por:
                img[i,j]=255
            elif random.random()>1-por:
                img[i,j]=0
    return img
img=cv.imread("./textjojo.jpg")
img_new=make_jiaoyanzaoyin(img,0.02)
plt.subplot(211)
plt.imshow(img_new[:,:,::-1])
plt.xticks([]),plt.yticks([])
plt.subplot(212)
plt.imshow(img_new)
plt.xticks([]),plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("./textjojo_new.jpg",img_new)


