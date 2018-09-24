import cv2 as cv
import numpy as np

im1 = cv.imread('/home/hamit/pcl-sw/libfreenect2pclgrabber/build/ir_1.png')
gray1=cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
ret,thresh1 = cv.threshold(gray1,200,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(gray1,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(gray1,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
cv.imshow("im1", im1)
cv.imshow("thresh1", thresh1)
cv.imshow("th2", th2)
cv.imshow("th3", th3)

cv.waitKey(0)
cv.destroyAllWindows()
