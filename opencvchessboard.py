import numpy as np
import cv2
import OpenEXR, Imath






# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


#First color and dept image
pt1 = Imath.PixelType(Imath.PixelType.FLOAT)
depth1File = OpenEXR.InputFile("/home/hamit/pcl-sw/libfreenect2pclgrabber/build/depth_0.exr")
dw1 = depth1File.header()['dataWindow']
size1 = (dw1.max.x - dw1.min.x + 1, dw1.max.y - dw1.min.y + 1)
depth1str = depth1File.channel('Y', pt1)
depth1 = np.fromstring(depth1str, dtype = np.float32)
depth1.shape = (size1[1], size1[0])

image1="/home/hamit/pcl-sw/libfreenect2pclgrabber/build/0.png"
img1 = cv2.imread(image1)
#img.convertTo(img, cv2.CV_32FC4);
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img11 = img1.copy()
# Find the chess board corners
#res1, corners1 = cv2.findChessboardCorners ( gray1, (7,5),cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
res1, corners1 = cv2.findChessboardCorners ( gray1, (7,5),None)

#ret, corners= cv2.findCirclesGrid(gray, (3,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

corners1Found=np.array([])

#Second color and dept image
pt2 = Imath.PixelType(Imath.PixelType.FLOAT)
depth2File = OpenEXR.InputFile("/home/hamit/pcl-sw/libfreenect2pclgrabber/build/depth_1.exr")
dw2 = depth1File.header()['dataWindow']
size2 = (dw2.max.x - dw2.min.x + 1, dw2.max.y - dw2.min.y + 1)
depth2str = depth2File.channel('Y', pt2)
depth2 = np.fromstring(depth2str, dtype = np.float32)
depth2.shape = (size2[1], size2[0])

image2="/home/hamit/pcl-sw/libfreenect2pclgrabber/build/1.png"
img2 = cv2.imread(image2)
#img.convertTo(img, cv2.CV_32FC4);
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,gray2 = cv2.threshold(gray2, 50, 255, 0);
img22 = img2.copy()
# Find the chess board corners
#res2, corners2 = cv2.findChessboardCorners ( gray2, (7,5),cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
res2, corners2 = cv2.findChessboardCorners ( gray2, (7,5),None)
#res2, corners2= cv2.findCirclesGrid(gray2, (3,11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

# If found, add object points, image points (after refining them)
corners2Found=np.array([])

cv2.imshow("gray2",gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()

if  res1== True and res2 == True:
  #  objpoints.append(objp)

    corners1Found = cv2.cornerSubPix(gray1,corners1,(11,11),(-1,-1),criteria)
    corners1FoundRev=corners1Found[::-1]
    #imgpoints.append(corners1)
    #print ("corners",corners2)
    #print ("cornersRev,", corners2Rev)

    # Draw and display the corners
    img1 = cv2.drawChessboardCorners(img1, (7,5), corners1Found,res1)
    cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img1', 900,900)
    #img11 = cv2.drawChessboardCorners(img1, (7,5), corners1Found,res1)
    cv2.imshow('img1',img1)
   # cv2.imshow('img11',img11)

    corners2Found = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
    corners2FoundRev=corners2Found[::-1]
    #imgpoints.append(corners1)
    #print ("corners",corners2)
    #print ("cornersRev,", corners2Rev)

    # Draw and display the corners
    img2 = cv2.drawChessboardCorners(img2, (7,5), corners2Found,res2)
    cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', 900,900)
    #img11 = cv2.drawChessboardCorners(img1, (7,5), corners1Found,res1)
    cv2.imshow('img2',img2)
    #cv2.imshow('img11',img11)

    while cv2.waitKey(0) != ord('q'):
        pass
    cv2.destroyAllWindows()

cx=959.5
cy=539.5
fx=1081.37
fy=1081.37
tmp_obj=[]

for tmp in corners1Found[:,0,::-1]:
    depthVal=depth1[int(np.round(tmp[0])),int(np.round(tmp[1]))]/1000.0
    print ("deptVal1=", depthVal,int(np.round(tmp[0])), int(np.round(tmp[1])))
    x= depthVal * (int(np.round(tmp[1]))-cx ) / fx
    y= depthVal * (int(np.round(tmp[0]))-cy ) / fy
    print ("Coord1 X, Y, Z ", x, y, depthVal)
    tmp_obj.append([x, y, depthVal] )

objpoints.append(tmp_obj.copy())

del tmp_obj[:]
a=corners2Found[:,0,::-1]
b=corners2Found[:,0,::-1].copy()
for i in range(0,0):
  b[i*7:(i+1)*7]=a[i*7:(i+1)*7][::-1]

for tmp in b:
    depthVal=depth2[int(np.round(tmp[0])),int(np.round(tmp[1]))]/1000.0
    print ("deptVal2=", depthVal,int(np.round(tmp[0])), int(np.round(tmp[1])))
    x= depthVal * (int(np.round(tmp[1]))-cx ) / fx
    y= depthVal * (int(np.round(tmp[0]))-cy ) / fy
    print ("Coord2 X, Y, Z ", x, y, depthVal)
    tmp_obj.append([x, y, depthVal])

objpoints.append(tmp_obj)
