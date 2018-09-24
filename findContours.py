import numpy as np
import cv2
from math import sqrt
import OpenEXR, Imath
import camParams as cam

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def colorDeptImage(colorFile, depthFile):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depthFile_ = OpenEXR.InputFile(depthFile)
    dw = depthFile_.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    depthstr = depthFile_.channel('Y', pt)
    depth = np.fromstring(depthstr, dtype = np.float32)
    depth.shape = (size[1], size[0])

    img = cv2.imread(colorFile)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res,thresh = cv2.threshold(imgray,120,255,0)
    img0, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=list(filter(lambda x: cv2.contourArea(x) > 1000 and  cv2.contourArea(x)<5000, contours ))


#First color and dept image
pt1 = Imath.PixelType(Imath.PixelType.FLOAT)
depth1File = OpenEXR.InputFile("/home/hamit/pcl-sw/libfreenect2pclgrabber/build/depth_0.exr")
dw1 = depth1File.header()['dataWindow']
size1 = (dw1.max.x - dw1.min.x + 1, dw1.max.y - dw1.min.y + 1)
depth1str = depth1File.channel('Y', pt1)
depth1 = np.fromstring(depth1str, dtype = np.float32)
depth1.shape = (size1[1], size1[0])

im1 = cv2.imread('/home/hamit/pcl-sw/libfreenect2pclgrabber/build/0.png')
imgray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
res1,thresh1 = cv2.threshold(imgray1,50,255,0)
th1 = cv2.adaptiveThreshold(imgray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
#imgray=cv2.blur( imgray, (3,3) )
im11, contours1, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print (len(contours1))

for contour in contours1:
    area = cv2.contourArea(contour)
    if area < 100  or area > 100000 :
        continue

    print (area)

#contours=[contour for contour in contours if cv2.contourArea(contour) >100 and cv2.contourArea(contour)<100000 ]
contours1=list(filter(lambda x: cv2.contourArea(x) > 1000 and  cv2.contourArea(x)<5000, contours1 ))
contours11=[]
for contour in contours1:
    epsilon = 0.01*cv2.arcLength(contour,True)
    corners=cv2.approxPolyDP(contour, 0.1*sqrt(cv2.contourArea(contour)), True)
#    corners1Found = cv2.cornerSubPix(im,corners,(5,5),(-1,-1),criteria)
    hull= cv2.convexHull(corners, clockwise=True)
    if  len(corners)==6 and len(hull)==5 :
        contours11.append(hull)

      #  print(cv2.contourArea(contour), cv2.arcLength(contour,True),corners)


#corners= list(map(lambda x: cv2.approxPolyDP(x, sqrt(cv2.contourArea(x))*0.12, True), contours)
#somelist[:] = ifilterfalse(determine, somelist)
#contours11=np.array(contours11)
#print (np.roll(contours3[0], -2, axis=0 ))
print ("Contour 1 \n",contours11[0])


#Second color and dept image
pt2 = Imath.PixelType(Imath.PixelType.FLOAT)
depth2File = OpenEXR.InputFile("/home/hamit/pcl-sw/libfreenect2pclgrabber/build/depth_1.exr")
dw2 = depth1File.header()['dataWindow']
size2 = (dw2.max.x - dw2.min.x + 1, dw2.max.y - dw2.min.y + 1)
depth2str = depth2File.channel('Y', pt2)
depth2 = np.fromstring(depth2str, dtype = np.float32)
depth2.shape = (size2[1], size2[0])

im2 = cv2.imread('/home/hamit/pcl-sw/libfreenect2pclgrabber/build/1.png')
imgray2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
res2,thresh2 = cv2.threshold(imgray2,100,255,0)
th2 = cv2.adaptiveThreshold(imgray2,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
# cv2.imshow("thresh2",thresh2)
# cv2.waitKey(0)
#imgray=cv2.blur( imgray, (3,3) )
im22, contours2, hierarchy2 = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print (len(contours2))

# for contour in contours2:
#     area = cv2.contourArea(contour)
#     if area < 100  or area > 100000 :
#         continue

    #print (area)

#contours=[contour for contour in contours if cv2.contourArea(contour) >100 and cv2.contourArea(contour)<100000 ]
contours2=list(filter(lambda x: cv2.contourArea(x) > 200 and  cv2.contourArea(x)<5000, contours2 ))
contours22=[]
for contour in contours2:
    epsilon = 0.01*cv2.arcLength(contour,True)
    corners=cv2.approxPolyDP(contour, 0.1*sqrt(cv2.contourArea(contour)), True)
#    corners1Found = cv2.cornerSubPix(im,corners,(5,5),(-1,-1),criteria)
    hull= cv2.convexHull(corners);
    if len(hull) == 5 and len(corners)==6:
        contours22.append(hull)

       # print(cv2.contourArea(contour), cv2.arcLength(contour,True),hull)


#corners= list(map(lambda x: cv2.approxPolyDP(x, sqrt(cv2.contourArea(x))*0.12, True), contours)
#somelist[:] = ifilterfalse(determine, somelist)
contours22=np.array(contours22)
contours22=np.roll(contours22[0], 0, axis=0 )
contours22=np.array([contours22])
print ( "Contour 2 \n",   contours22[0])
#print ( "Contour 2 \n",   contours22[0])

#print (contours22)





cv2.drawContours(im1, contours11, -1, (0, 255, 0), 1)
cv2.drawContours(im2, contours22, -1, (0, 255, 0), 1)
cv2.imshow("im2", im2)
cv2.imshow("im1", im1)
cv2.imshow("thresh2",thresh2)
cv2.imshow("thresh1",thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()


objpoints = []

cx,cy,fx,fy=cam.cam021401644747()

tmp_obj=[]

for tmp in contours11[0]:
    depthVal=depth1[int(np.round(tmp[0][1])),int(np.round(tmp[0][0]))]/1000.0
    print ("deptVal1=", depthVal,int(np.round(tmp[0][1])), int(np.round(tmp[0][0])))
    x= depthVal * (int(np.round(tmp[0][0]))-cx + 0.5 ) / fx
    y= depthVal * (int(np.round(tmp[0][1]))-cy + 0.5 ) / fy
    print ("Coord1 X, Y, Z ", x, y, depthVal)
    tmp_obj.append([x, y, depthVal] )

objpoints.append(tmp_obj.copy())

del tmp_obj[:]



cx,cy,fx,fy=cam.cam291296634347()


#contours22Ord= np.concatenate((np.array([contours22[0][0]]),contours22[0][1:][::-1]),axis=0)
#print ( "Contour 2 \n",   contours22Ord)
for tmp in contours22[0]:
    depthVal=depth2[int(np.round(tmp[0][1])),int(np.round(tmp[0][0]))]/1000.0
    print ("deptVal2=", depthVal,int(np.round(tmp[0][1])), int(np.round(tmp[0][0])))
    x= depthVal * (int(np.round(tmp[0][0]))-cx +0.5) / fx
    y= depthVal * (int(np.round(tmp[0][1]))-cy +0.5) / fy
    print ("Coord2 X, Y, Z ", x, y, depthVal)
    tmp_obj.append([x, y, depthVal])

objpoints.append(tmp_obj)


