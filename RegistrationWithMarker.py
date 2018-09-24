#!/home/hamit/anaconda2/bin/python
import numpy as np
import cv2
from math import sqrt
import OpenEXR, Imath
#import camParams as cam

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class camera:
    def __init__(self, camserial):
        if camserial == "021401644747":
            self.cx=2.59602295e+02
            self.cy=2.04101303e+02
            self.fx=3.67825592e+02
            self.fy=3.67825592e+02
        elif camserial == "291296634347":
            self.cx=2.57486603e+02
            self.cy=1.99466003e+02
            self.fx=3.64963013e+02
            self.fy=3.64963013e+02
        elif camserial == "003726334247":
            self.cx=2.54701508e+02
            self.cy=2.01162506e+02
            self.fx=3.64941895e+02
            self.fy=3.64941895e+02
        else:
            print("It's not defined camera serial")

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
   # H = BB * transpose(AA)
    H = np.transpose(AA) * BB
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T
    #R = U * Vt
    # special reflection case
    if np.linalg.det(R) < 0:
        print ("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T



    return R, t


def nothing(x):
    pass

def loadColorDepthImage(colorFile, depthFile):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    depthFile_ = OpenEXR.InputFile(depthFile)
    dw = depthFile_.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    depthstr = depthFile_.channel('Y', pt)
    depth = np.fromstring(depthstr, dtype = np.float32)
    depth.shape = (size[1], size[0])
    img = cv2.imread(colorFile)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img,imgray, depth

def createContour(imgray, grayScale, clockwise=True):

    res,thresh = cv2.threshold(imgray,grayScale,255,0)
    img0, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=list(filter(lambda x: cv2.contourArea(x) > 500 and  cv2.contourArea(x)<5000, contours ))

    contours0 = []
    for contour in contours:
        #print ("cv2.contourArea(contour)", cv2.contourArea(contour))
        epsilon = 0.01 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.1 * sqrt(cv2.contourArea(contour)), True)
        #    corners1Found = cv2.cornerSubPix(im,corners,(5,5),(-1,-1),criteria)
        hull = cv2.convexHull(corners, clockwise=clockwise)
        if len(corners) == 8 and len(hull) == 6:
            #print ("cv2.contourArea(contour)", cv2.contourArea(contour))
            contoursDist=[]
            if clockwise==True:
                contoursDist= list(map(lambda x:x[0][0]**2 + x[0][1]**2, hull))
            else:
                contoursDist= list(map(lambda x:(x[0][0]-imgray.shape[1])**2 + x[0][1]**2, hull))

            #print ("contourdist ", contoursDist)
            index=contoursDist.index(min(contoursDist))
            #print ("index: {} ".format(index))
            hull=np.roll(hull, -index, axis=0 )
            contours0.append(hull)



    return contours0

def computeCoords(contour, depth, cam):
    tmp_obj=[]

    try:
        for tmp in contour[0]:
            depthVal=depth[int(np.round(tmp[0][1])),int(np.round(tmp[0][0]))]/1000.0
            #print ("deptVal1=", depthVal,int(np.round(tmp[0][1])), int(np.round(tmp[0][0])))
            x= depthVal * (int(np.round(tmp[0][0]))-cam.cx + 0.5 ) / cam.fx
            y= depthVal * (int(np.round(tmp[0][1]))-cam.cy + 0.5 ) / cam.fy
           # print ("Coord1 X, Y, Z ", x, y, depthVal)
            tmp_obj.append([x, y, depthVal] )
        return tmp_obj
    except IndexError:
        print("No contour selected !!!")
        return tmp_obj

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="registrationMarker")
    parser.add_argument("-f1", "--files1", dest="fileNames1", required=True,
                        help="1st file ", metavar="FILE")
    parser.add_argument("-f2", "--files2", dest="fileNames2",required=True,
                        help="2nd file ", metavar="FILE")
    args = parser.parse_args()

    contoursList=[]
    clockwise=False
    objpoints=[]

    #for arg in vars(args):
    for key, value in sorted(vars(args).items()):
       # print (arg, getattr(args, arg))
       # print ("key value", key, value)
        imgFile=value
        if imgFile is None:
            continue
        exrFile = imgFile.split('.')[0] + ".exr"
       # print("file: ",files[0].split('.')[0])
        img, imgray, depth = loadColorDepthImage(imgFile, exrFile)

       # img, imgray, depth = loadColorDepthImage("/home/hamit/pcl-sw/libfreenect2pclgrabber/build/0.png","/home/hamit/pcl-sw/libfreenect2pclgrabber/build/depth_0.exr")

        #cv2.cvtColor(input, input_bgra, CV_BGR2BGRA);

        windowName="TrackbarGrayScale"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 900,900)

        mask = np.zeros(img.shape, np.uint8)
       # mask = np.ones(img.shape[:2], dtype="uint8") * 255
        cv2.createTrackbar( 'GrayScale', windowName, 0, 255, nothing)



        contours_found=True

        clockwise= not clockwise

        while(1):

            imgcp=img.copy()

            grayScale = cv2.getTrackbarPos('GrayScale',windowName)

            contours = createContour(imgray, grayScale, clockwise )


            if len(contours) > 0:
                cv2.drawContours(imgcp, contours, -1, (0, 255, 0), 1)
                contours_found=True
                for contour in contours[0]:
                   # print ("contour:", contour[0])
                    cv2.circle(imgcp, tuple(contour[0]), 1, (0, 0, 255) ,  -1);
                cv2.imshow("TrackbarGrayScale", imgcp)
            elif len(contours) == 0 and contours_found :
                cv2.drawContours(mask, contours,-1, (255,255,255,255), 1)

               # img = cv2.bitwise_and(img, img, mask=mask)
                removed = cv2.add(imgcp, mask)
                cv2.imshow("TrackbarGrayScale", removed)

            else:
                cv2.imshow("TrackbarGrayScale", imgcp)




            imgcp=[]


            if cv2.waitKey(1) == ord('q') :

                tmpcamStr=imgFile.split('.')[0].split('/')[-1].split('_')[0]
                #print ("tmpcamstr", tmpcamStr)

                cont=computeCoords(contours,depth, camera(tmpcamStr))

                if len(contours) < 1:
                      continue
                objpoints.append(cont)
                contoursList.append([contours,  imgFile.split('.')[0].split('/')[-1]])

                break

        cv2.destroyAllWindows()

    print ("ContoursList \n", contoursList)

    A=np.asmatrix(objpoints[1])
    B=np.asmatrix(objpoints[0])

    ret_R, ret_t = rigid_transform_3D(A, B)

    # A2 = (ret_R * A.T) + np.tile(ret_t, (1, n))
    # A2 = A2.T
    #
    # # Find the error
    # err = A2 - B
    #
    # err = np.multiply(err, err)
    # err = np.sum(err)
    # rmse = sqrt(err / n)


    print ("Rotation")
    print (ret_R)
    print ("")

    print ("Translation")
    print (ret_t)

    trans= np.vstack([np.hstack((ret_R,ret_t)),[0,0,0,1]])
    transOutputFile=""
    for increment,arg in enumerate(sorted(vars(args))):
        a=getattr(args, arg)
        a=a.split('.')[0].split('/')[-1]
        transOutputFile+=a
        if increment  < len(vars(args))-1:
            transOutputFile+="_to_"

        #print ("fileName: ",transOutputFile)

    np.savetxt(transOutputFile+'.out', trans , delimiter=' ',fmt='%1.8f')
