#!/usr/bin/python
import numpy as np
import cv2
from math import sqrt
from math import log
from math import exp
#import ROOT
# from ROOT import *
from array import array
import pdb
import math
from copy import deepcopy
import subprocess
import os, glob, sys

# import OpenEXR, Imath
# import camParams as cam

A = np.matrix([])
B = np.matrix([])
prevfval = 0.02


def euler2mat(z=0, y=0, x=0):
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def eulerAnglesToRotationMatrix(theta=[0., 0., 0.]):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    # R= np.matrix(R_z)* np.matrix(R_y)* np.matrix(R_x)
    return R


def fun(par):
    rot = np.matrix(par)
    rot = rot.reshape(4, 4)
    global A, B
    C = np.append(A, np.ones((A.shape[0], 1)), axis=1)
    C = np.transpose(C)
    D = rot * C
    print(D, C, B)



##______________________________________________________________________________
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class camera:
    def __init__(self, camserial):
        if camserial == "011054343347":
            self.cx = 2.60037109e+02
            self.cy = 2.07033997e+02
            self.fx = 3.66541595e+02
            self.fy = 3.66541595e+02
        elif camserial == "291296634347":
            self.cx = 2.57486603e+02
            self.cy = 1.99466003e+02
            self.fx = 3.64963013e+02
            self.fy = 3.64963013e+02
        elif camserial == "003726334247":
            self.cx = 2.54701508e+02
            self.cy = 2.01162506e+02
            self.fx = 3.64941895e+02
            self.fy = 3.64941895e+02
        elif camserial == "097663440347":
            self.cx = 2.63912109e+02
            self.cy = 2.03802994e+02
            self.fx = 3.65381500e+02
            self.fy = 3.65381500e+02
        elif camserial == "021401644747":
            self.cx = 2.59602295e+02
            self.cy = 2.04101303e+02
            self.fx = 3.67825592e+02
            self.fy = 3.67825592e+02
        elif camserial == "127185440847":
            self.cx = 2.54904007e+02
            self.cy = 2.02395905e+02
            self.fx = 3.66524414e+02
            self.fy = 3.66524414e+02
        elif  camserial =="074131541147":
            self.cx = 254.6179962158203
            self.cy = 203.18099975585938
            self.fx = 365.810302734375
            self.fy = 365.810302734375
        elif camserial=="119943135047":
            self.cx = 261.5863952636719
            self.cy = 204.70249938964844
            self.fx = 366.2644958496094
            self.fy = 366.2644958496094
        elif camserial=="156217635047":
            self.cx = 254.19110107421875
            self.cy = 205.03790283203125
            self.fx = 366.101806640625
            self.fy = 366.101806640625
        elif camserial=="060004434547":
            self.cx =261.07830810546875
            self.cy = 202.277099609375
            self.fx = 365.993408203125
            self.fy = 365.993408203125

        else:
            print("camera serial {} is not defined ".format(camera))


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
    #print("sss ", S)
    R = Vt.T * U.T
    # R = U * Vt
    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    return R, t


def nothing(x):
    pass


def loadColorDepthImage(colorFile, depthFile):
    depth = cv2.imread(depthFile, cv2.IMREAD_ANYDEPTH)
    img = cv2.imread(colorFile)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, imgray, depth


def createContour(imgray, depth, grayScale, clockwise=True):
    res, thresh = cv2.threshold(imgray, grayScale, 255, 0)
    img0, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: cv2.contourArea(x) > 500 and cv2.contourArea(x) < 40000, contours))

    contours0 = []
    found=False;
    for contour in contours:
        # print ("cv2.contourArea(contour)", cv2.contourArea(contour))
        epsilon = 0.01 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.05 * sqrt(cv2.contourArea(contour)), True)

        #    corners1Found = cv2.cornerSubPix(im,corners,(5,5),(-1,-1),criteria)
        hull = cv2.convexHull(corners, clockwise=clockwise)
        # print (len(corners), len(hull))

        if len(corners) == 18 and len(hull) == 11:

            found=True

            for corner in corners:
                #print("corners", corner)
                depthVal = depth[int(np.round(corner[0][1])), int(np.round(corner[0][0]))] / 1000.0
                if depthVal > 3 :
                    found=False
                    return contours0, found

            # if len(corners) == 10 and len(hull) == 7:
            # print ("cv2.contourArea(contour)", cv2.contourArea(contour))
            contoursDist = []
            if clockwise == True:
                contoursDist = list(map(lambda x: x[0][0] ** 2 + x[0][1] ** 2, hull))
            else:
                contoursDist = list(map(lambda x: (x[0][0] - imgray.shape[1]) ** 2 + x[0][1] ** 2, hull))

            # print ("contourdist ", contoursDist)
            index = contoursDist.index(min(contoursDist))
            # print ("index: {} ".format(index))
            hull = np.roll(hull, -index, axis=0)
            contours0.append(hull)

    return contours0, found


def computeCoords(contour, depth, cam):
    tmp_obj = []

    try:
        for tmp in contour[0]:
            depthVal = depth[int(np.round(tmp[0][1])), int(np.round(tmp[0][0]))] / 1000.0
            # print ("deptVal1=", depthVal,int(np.round(tmp[0][1])), int(np.round(tmp[0][0])))
            x = depthVal * (int(np.round(tmp[0][0])) - cam.cx + 0.5) / cam.fx
            y = depthVal * (int(np.round(tmp[0][1])) - cam.cy + 0.5) / cam.fy
            xpl = depthVal * (int(np.round(tmp[0][0] + 4)) - cam.cx + 0.5) / cam.fx,
            ypl = depthVal * (int(np.round(tmp[0][1] + 4)) - cam.cy + 0.5) / cam.fy
            # x-xpl, y-ypl
            errdepth = 0.04
            errx = errdepth * (int(np.round(tmp[0][0])) - cam.cx + 0.5) / cam.fx
            erry = errdepth * (int(np.round(tmp[0][1])) - cam.cy + 0.5) / cam.fy
            toterrx = sqrt(pow(errx, 2) + pow(x - xpl, 2))
            toterry = sqrt(pow(erry, 2) + pow(y - ypl, 2))

           # print("error X, T ", x, y, depthVal)
            # print ("Error on X, Y, Z ", toterrx, toterry, errdepth)
            # tmp_obj.append([x, y, depthVal] )
            tmp_obj.append([[x, y, depthVal], [toterrx, toterry, errdepth]])
        return tmp_obj
    except IndexError:
        print("No contour selected !!!")
        return tmp_obj


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="registrationMarker")
    parser.add_argument("-s", "--serialFile", dest="serialsFile", required=True,
                        help="Kinect Serials File", metavar="SERIALS")

    args = parser.parse_args()

    #  print sorted(vars(args).items())

    serialsListContourRot = []
    # for key, value in sorted(vars(args).items()):
    #     if key is "serials":
    #         serialsList=value

    # print ("serials ", serialsList)

    with open(args.serialsFile, "r") as infile:
        for line in infile:
            serialsListContourRot.append((line.split()[0], line.split()[1]))

    print("serials ", serialsListContourRot)
    referenceSerial = ""
    for k, value in enumerate(serialsListContourRot):

        if (k == 0):
            referenceSerial = value[0]
            continue
        # else:
        #      referenceSerial= serialsListContourRot[k-1][0]
        os.system("rm *.png")
        os.system(
            "/home/hamit/libfreenect2pclgrabber/devel/lib/kinect2grabber/rosPairKinectv2Viewer2  --serials " + referenceSerial + " " + value[0]
         )



        contoursList = []
        clockwise0 = True
        clockwise1 = True
        objpoints = []


        pngFiles0=glob.glob(referenceSerial+"*_0.png")
        pngFiles0.sort(key=os.path.getmtime)
        depthFiles0=list(map (lambda x: x.split('.')[0] + "_depth.png", pngFiles0))

        pngFiles1=glob.glob(value[0]+"*_1.png")
        pngFiles1.sort(key=os.path.getmtime)
        depthFiles1=list(map (lambda x: x.split('.')[0] + "_depth.png", pngFiles1))

        if len(pngFiles0)!= len(pngFiles1):
            print("Number of files of each cam is not the same, moving on to another set of cams! ....")

            continue



        allcont0=[]
        allcont1=[]
        for j in range (0, len(depthFiles0) ):

            img0, imgray0, depth0 = loadColorDepthImage(pngFiles0[j], depthFiles0[j])

            contours0=[]
            isFound0=False
            grayScale0=0


            for scl in range (0, 256 ):

                contours0, isFound0 = createContour(imgray0, depth0, scl, clockwise0)
                if isFound0 ==True:
                    grayScale0=scl
                    break

            if isFound0==False:
                print("Any contour was not found, exiting! ....")
                continue


            windowName0="TrackbarGrayScale0"
            cv2.namedWindow(windowName0, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName0, 900,900)

            mask0 = np.zeros(img0.shape, np.uint8)
            cv2.createTrackbar( 'GrayScale0', windowName0, 0, 255, nothing)
            cv2.setTrackbarPos('GrayScale0', windowName0, grayScale0)



            img1, imgray1, depth1 = loadColorDepthImage(pngFiles1[j], depthFiles1[j])

            contours1=[]
            isFound1=False
            grayScale1=0

            if ( bool(int(value[1]) == True)):
                    clockwise1= not clockwise1

            for scl in range (0, 256 ):

                contours1, isFound1 = createContour(imgray1, depth1, scl, clockwise1)
                if isFound1 ==True:
                    grayScale1=scl
                    continue

            windowName1="TrackbarGrayScale1"
            cv2.namedWindow(windowName1, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName1, 900,900)
            cv2.moveWindow(windowName1, 900,0)

            mask1 = np.zeros(img1.shape, np.uint8)
            cv2.createTrackbar( 'GrayScale1', windowName1, 0, 255, nothing)
            cv2.setTrackbarPos('GrayScale1', windowName1, grayScale1)



            isbreak=False

            # c=cv2.waitKey()
            #cv2.destroyAllWindows()

            # if (c==ord('t')):



            while(1):

                imgcp0=img0.copy()



                grayScale0 = cv2.getTrackbarPos('GrayScale0',windowName0)
                contours0, isFound0 = createContour(imgray0, depth0, grayScale0, clockwise0 )


                if  isFound0==True:
                    cv2.drawContours(imgcp0, contours0, -1, (0, 255, 0), 1)
                    contours_found=True
                    for contour in contours0[0]:
                        cv2.circle(imgcp0, tuple(contour[0]), 1, (0, 0, 255) ,  -1);
                    cv2.imshow(windowName0, imgcp0)
                elif len(contours0) == 0 and isFound0 :
                    cv2.drawContours(mask0, contours0,-1, (255,255,255,255), 1)

                    removed = cv2.add(imgcp0, mask0)
                    cv2.imshow(windowName0, removed)

                else:
                    cv2.imshow(windowName0, imgcp0)


                imgcp0=[]

                imgcp1=img1.copy()



                grayScale1 = cv2.getTrackbarPos('GrayScale1',windowName1)
                contours1, isFound1 = createContour(imgray1, depth1, grayScale1, clockwise1 )


                if  isFound1==True:
                    cv2.drawContours(imgcp1, contours1, -1, (0, 255, 0), 1)
                    contours_found=True
                    for contour in contours1[0]:
                        cv2.circle(imgcp1, tuple(contour[0]), 1, (0, 0, 255) ,  -1);
                    cv2.imshow(windowName1, imgcp1)
                elif len(contours1) == 0 and isFound1 :
                    cv2.drawContours(mask1, contours1,-1, (255,255,255,255), 1)

                    removed = cv2.add(imgcp1, mask1)
                    cv2.imshow(windowName1, removed)

                else:
                    cv2.imshow(windowName1, imgcp1)

                imgcp1=[]


                c=cv2.waitKey(2)

                if c == ord('q') :
                    isbreak=True
                    break
                elif c == ord('s') or c == ord(' ') :
                    break
                # if c == ord('q') :
                #     isbreak=True
                #     break





            if isbreak is True:
                cv2.destroyAllWindows()
                continue
            cv2.destroyAllWindows()




            tmpcamStr0 = pngFiles0[j].split('.')[0].split('/')[-1].split('_')[0]
            cont0 = computeCoords(contours0, depth0, camera(tmpcamStr0))
            allcont0.extend(cont0)
            tmpcamStr1 = pngFiles1[j].split('.')[0].split('/')[-1].split('_')[0]
            cont1 = computeCoords(contours1, depth1, camera(tmpcamStr1))
            allcont1.extend(cont1)

        objpoints.append(allcont0)
        objpoints.append(allcont1)














       # print("ContoursList \n", contoursList)
        objpointsNp = np.array(objpoints)
        print (len(objpointsNp), objpointsNp.shape[1])
        if len(objpointsNp[0])==0  or len(objpointsNp[1])==0:
            continue
        A = np.asmatrix(objpointsNp[1][:, 0])
        B = np.asmatrix(objpointsNp[0][:, 0])
        Aerr = np.asmatrix(objpointsNp[1][:, 1])
        Berr = np.asmatrix(objpointsNp[0][:, 1])
       # print('{0} and \n {1}'.format(Aerr, Berr))
        # printf("A,\n B")


        ret_R, ret_t = rigid_transform_3D(A, B)

        #A2 = (ret_R * A.T) + np.tile(ret_t, (1, objpointsNp.shape[1]))
        # A2 = A2.T
        #
        # # Find the error
        #err = A2.T - B
        #
        #print("Error:", err)
        #err = np.multiply(err, err)
        #err = np.sum(err)
        #rmse = sqrt(err /)

        print("Rotation")
        print(ret_R)
        print("")

        print("Translation")
        print(ret_t)

        trans = np.vstack([np.hstack((ret_R, ret_t)), [0, 0, 0, 1]])
        transOutputFile = ""
        for increment, arg in enumerate([referenceSerial, value[0]]):

            a = arg.split('.')[0].split('/')[-1]
            transOutputFile += a
            if increment < 1:
                transOutputFile += "_to_"

            # print ("fileName: ",transOutputFile)

        np.savetxt(transOutputFile + '.out', trans, delimiter=' ', fmt='%1.8f')
        np.savetxt(str(k) + '.out', trans, delimiter=' ', fmt='%1.8f')