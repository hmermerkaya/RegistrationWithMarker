#!/home/hamit/anaconda2/bin/python
import numpy as np
import cv2
from math import sqrt
from math import log
from math import exp
import ROOT
#from ROOT import *
from array import array
import pdb
import math
from copy import deepcopy
import subprocess
import os, sys

#import OpenEXR, Imath
#import camParams as cam

A=np.matrix([])
B=np.matrix([])
prevfval=0.02

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

def eulerAnglesToRotationMatrix(theta=[0., 0., 0.]) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,                        0 ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))
   # R= np.matrix(R_z)* np.matrix(R_y)* np.matrix(R_x)
    return R



def fun(par):
    rot=np.matrix(par)
    rot=rot.reshape(4,4)
    global A,B
    C=np.append(A, np.ones((A.shape[0],1)), axis=1)
    C=np.transpose(C)
    D=rot*C
    print(D,C,B)





def fcnchi2(npar, gin, f, par, iflag):


    global A,B, Aerr, prevfval
    rot= np.matrix(eulerAnglesToRotationMatrix(theta=[par[0], par[1], par[2]]))
    #rot = np.matrix(euler2mat(par[0], par[1], par[2]))
    transl0=np.matrix([[par[3]],[par[4]],[par[5]]])
    trans=np.append(rot,transl0, axis=1)
    transl1=np.matrix([[0,0,0,1]])
    transfinal=np.append(trans,transl1, axis=0)



    #print transfinal
    AA = np.append(A, np.ones((A.shape[0],1)), axis=1)
    AA = np.transpose(AA)
    BB = np.append(B, np.ones((B.shape[0],1)), axis=1)
    BB = np.transpose(BB)
   # print('{0} times \n {1} is {2} '.format(rot,AA, rot*AA))
    #print( BB[0:-1],AA, rot*AA)
    #print(BB, "\n", rot*AA)
    deltaAandB = np.subtract((transfinal*AA)[0:-1], BB[0:-1])
    sqrtB=np.sqrt(np.abs(np.divide(BB[0:-1],10)))
    #print(AA[0,-1])
    #print ('{0} and\n {1}'.format(deltaAandB, np.transpose(Aerr)))
    chi2AandB = np.divide(deltaAandB, np.transpose(Berr))
   # print ('chi2 \n {0}'.format(chi2AandB))
    #chi2SquareAandB=np.square(chi2AandB)
    chi2SquareAandB=np.square(deltaAandB)



    f[0] = np.sum(chi2SquareAandB)

def fcnlnL(npar, gin, f, par, iflag):
    # z = np.array((1.0, 0.96, 0.89, 0.85, 0.78))
    # errorz = np.array((0.01, 0.01, 0.01, 0.01, 0.01))
    # x = np.array((1.5751, 1.5825, 1.6069, 1.6339, 1.6706))
    # y = np.array((1.0642, 0.97685, 1.13168, 1.128654, 1.44016))

    # print "Nfree = ", npar[0], " Value of 4th Parm = ", par[3]
    # nbins = 5


    # calculate chisquare
    global A,B,Aerr

    rot=np.matrix([[par[0], par[1], par[2], par[3]],
                   [par[4], par[5], par[6], par[7]],
                   [par[8], par[9], par[10], par[11]],
                   [par[12], par[13], par[14], par[15]]
                ])

    AA = np.append(A, np.ones((A.shape[0],1)), axis=1)
    AA = np.transpose(AA)
   # BB = np.append(B, np.ones((B.shape[0],1)), axis=1)
   # BB = np.transpose(BB)


   # BB = np.transpose(BB)
   # print('{0} times \n {1} is {2} '.format(rot,AA, rot*AA))
   # Aerr =
    #print( rot,AA, rot*AA)
    #print(BB, "\n", rot*AA)
       # value = ((par[0] * par[0]) / (xx * xx) - 1) / (par[1] + par[2] * yy - par[3] * yy * yy);
    #deltaAandB = np.subtract((rot*AA)[0:-1], BB[0:-1])
   # BB = BB[0:-1]
    BBtrans =np.transpose((rot*AA)[0:-1])
    lnL = 0.0
    print (B.shape, BBtrans.shape)
    for (r,c), value in np.ndenumerate(B):
        lnL += BBtrans[r,c] - value + value * log(value/BBtrans[r,c])


    # for i,j in enumerate(B):
    #     print j
    #     for l, k in enumerate(j[0,:]):
    #        #lnL -= pow(BBtrans[i,l],k)*exp(-BBtrans[i,l])
    #        print k, l
           # lnL += BBtrans[i,l] - k + k * log(k/BBtrans[i,l])
           # print BBtrans[i,l]
    # for i,j in enumerate(B):
    #     print j,A[i]

    #print(AA[0,-1])
    #print ('{0} and\n {1}'.format(deltaAandB, np.transpose(Aerr)))
   # chi2AandB = np.divide(deltaAandB, np.transpose(Aerr))
   # print ('chi2 \n {0}'.format(chi2AandB))
    #chi2SquareAandB=np.square(chi2AandB)
    #chi2SquareAandB=np.square(deltaAandB)



    f[0] = 5 #lnL; np.sum(chi2SquareAandB)
def Ifit2():
    min=ROOT.TVirtualFitter.Fitter(0,6)
    min.SetFCN(fcnchi2)
    arglist = array('d', 10 * [0.])
    ierflg = ROOT.Long(1982)

    arglist[0] = 1
    min.ExecuteCommand("SET PRINT",arglist,ierflg)
    vstart = np.array([0.5, 0.5, 0.5,
                       0.,  0.,  0,
                       ])
    step = np.array([0.0001, 0.0001, 0.0001,
                     0.0001, 0.0001, 0.0001
                     ])
    min.SetParameter(0, "a0", vstart[0], step[0], -math.pi, math.pi)
    min.SetParameter(1, "a1", vstart[1], step[1], -math.pi, math.pi)
    min.SetParameter(2, "a2", vstart[2], step[2], -math.pi, math.pi)
    min.SetParameter(3, "a3", vstart[3], step[3], -10, 10)
    min.SetParameter(4, "a4", vstart[4], step[4], -10, 10)
    min.SetParameter(5, "a5", vstart[5], step[5], -10, 10)

    arglist[0] = ROOT.Long(50000000)
    arglist[1] = ROOT.double(0.00001)
    min.ExecuteCommand("MIGRAD", arglist, 2)
    min.ExecuteCommand("HESSE", arglist, 2)
    #Print results
    amin, edm, errdef = ROOT.Double(0.18), ROOT.Double(0.19), ROOT.Double(0.20)
    nvpar, nparx, icstat = ROOT.Long(1983), ROOT.Long(1984), ROOT.Long(1985)
    min.GetStats(amin, edm, errdef, nvpar, nparx)
    #aMinuit.mnprin(3, amin)
    print ("amin: ",amin)

    par =  ROOT.Double()
    error= ROOT.Double()
    parlist=[]
    for i in range(6):
        par0=min.GetParameter(i)
        parlist.append(deepcopy(par0))


    #print parlist
    print ('Rot Mat')
    ret_R = eulerAnglesToRotationMatrix(parlist[0:3])
    ret_t= np.transpose(np.matrix(parlist[3:]))
    #print (np.matrix(ret_R), np.matrix(ret_t))
    print('Rot TvirturalFitter \n{0}  \n and translation   \n {1}'.format(np.matrix(ret_R), np.matrix(ret_t)))

    #print 'Translation'

    #print ret_t
    print ("")

    print (min.GetNumberFreeParameters (), min.GetNumberTotalParameters ())

    trans= np.vstack([np.hstack((ret_R,ret_t)),[0,0,0,1]])
    np.savetxt("tranformation"+'.out', trans , delimiter=' ',fmt='%1.8f')

def Ifit():
    aMinuit = ROOT.TMinuit(6)
    aMinuit.SetFCN(fcnchi2)

    arglist = array('d', 10 * [0.])
    ierflg = ROOT.Long(1982)

    arglist[0] = 1
    aMinuit.mnexcm("SET ERR", arglist, 1, ierflg);

    # Set starting values and step sizes for parameters
    # vstart = np.array([0.5, 0.1, 0.1, 1,
    #                    0.1, 0.5, 0.1, 1,
    #                    0.1, 0.1, 0.5, 1,
    #                    0,   0,   0,   1 ])
    # step = np.array([0.001, 0.001, 0.001, 0.001,
    #                  0.001, 0.001, 0.001, 0.001,
    #                  0.001, 0.001, 0.001, 0.001,
    #                  0.001, 0.001, 0.001, 0.001
    #                  ])

    vstart = np.array([0.5, 0.5, 0.5,
                       1.,  1.,  1.
                       ])
    step = np.array([0.00001, 0.00001, 0.00001,
                     0.00001, 0.00001, 0.00001
                     ])

    aMinuit.mnparm(0, "a0", vstart[0], step[0], -math.pi, math.pi, ierflg)
    aMinuit.mnparm(1, "a1", vstart[1], step[1], -math.pi, math.pi, ierflg)
    aMinuit.mnparm(2, "a2", vstart[2], step[2], -math.pi, math.pi, ierflg)
    aMinuit.mnparm(3, "a3", vstart[3], step[3], -10, 10, ierflg)

    aMinuit.mnparm(4, "a4", vstart[4], step[4], -10, 10, ierflg)
    aMinuit.mnparm(5, "a5", vstart[5], step[5], -10, 10, ierflg)
    # aMinuit.mnparm(6, "a6", vstart[6], step[6], -1, 1, ierflg)
    # aMinuit.mnparm(7, "a7", vstart[7], step[7], -10, 10, ierflg)
    #
    # aMinuit.mnparm(8, "a8", vstart[8], step[8], -1, 1, ierflg)
    # aMinuit.mnparm(9, "a9", vstart[9], step[9], -1, 1, ierflg)
    # aMinuit.mnparm(10, "a10", vstart[10], step[10], -1, 1, ierflg)
    # aMinuit.mnparm(11, "a11", vstart[11], step[11], -10, 10, ierflg)
    #
    # aMinuit.mnparm(12, "a12", vstart[12], step[12], -1, 1, ierflg)
    # aMinuit.mnparm(13, "a13", vstart[13], step[13], -1, 1, ierflg)
    # aMinuit.mnparm(14, "a14", vstart[14], step[14], -1, 1, ierflg)
    # aMinuit.mnparm(15, "a15", vstart[15], step[15], -10, 10, ierflg)
    #
    # aMinuit.FixParameter(12);
    # aMinuit.FixParameter(13);
    # aMinuit.FixParameter(14);
    # aMinuit.FixParameter(15);
    # # fix one parameter
    # if (fixit):
    #     aMinuit.FixParameter(3)

    # Now ready for minimization step
    arglist[0] = ROOT.Long(50000000)
    arglist[1] = ROOT.double(0.00001)
    aMinuit.mnexcm("MINOS", arglist, 2, ierflg)

    #Print results
    amin, edm, errdef = ROOT.Double(0.18), ROOT.Double(0.19), ROOT.Double(0.20)
    nvpar, nparx, icstat = ROOT.Long(1983), ROOT.Long(1984), ROOT.Long(1985)
    aMinuit.mnstat(amin, edm, errdef, nvpar, nparx, icstat)
    #aMinuit.mnprin(3, amin)
    print ("amin: ",amin)

    par =  ROOT.Double()
    error= ROOT.Double()
    parlist=[]
    for i in range(6):
        aMinuit.GetParameter(i, par, error)
        parlist.append(deepcopy(par))


    #print parlist
    print ('Rot Mat')
    ret_R = eulerAnglesToRotationMatrix(parlist[0:3])
    ret_t= np.transpose(np.matrix(parlist[3:]))
    #print (np.matrix(ret_R), np.matrix(ret_t))
    print('Rot \n{0}  \n and translation   \n {1}'.format(np.matrix(ret_R), np.matrix(ret_t)))

    #print 'Translation'

    #print ret_t
    print ("")

    print (aMinuit.GetNumPars (), aMinuit.GetNumFreePars ())

    trans= np.vstack([np.hstack((ret_R,ret_t)),[0,0,0,1]])
    np.savetxt("tranformation"+'.out', trans , delimiter=' ',fmt='%1.8f')


##______________________________________________________________________________
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class camera:
    def __init__(self, camserial):
        if camserial == "011054343347":
            self.cx=2.60037109e+02
            self.cy=2.07033997e+02
            self.fx=3.66541595e+02
            self.fy=3.66541595e+02
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
        elif camserial == "097663440347":
            self.cx=2.63912109e+02
            self.cy=2.03802994e+02
            self.fx=3.65381500e+02
            self.fy=3.65381500e+02
        elif camserial == "021401644747":
            self.cx= 2.59602295e+02
            self.cy=  2.04101303e+02
            self.fx=  3.67825592e+02
            self.fy= 3.67825592e+02
        elif camserial == "127185440847":
            self.cx= 2.54904007e+02
            self.cy=  2.02395905e+02
            self.fx=  3.66524414e+02
            self.fy=  3.66524414e+02

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
    print("sss ",S)
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
    depth = cv2.imread(depthFile,cv2.IMREAD_ANYDEPTH)
    img = cv2.imread(colorFile)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img,imgray, depth



def createContour(imgray, grayScale, clockwise=True):

    res,thresh = cv2.threshold(imgray,grayScale,255,0)
    img0, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=list(filter(lambda x: cv2.contourArea(x) > 500 and  cv2.contourArea(x)<40000, contours ))

    contours0 = []
    for contour in contours:
        #print ("cv2.contourArea(contour)", cv2.contourArea(contour))
        epsilon = 0.01 * cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.05 * sqrt(cv2.contourArea(contour)), True)

        #    corners1Found = cv2.cornerSubPix(im,corners,(5,5),(-1,-1),criteria)
        hull = cv2.convexHull(corners, clockwise=clockwise)
        #print (len(corners), len(hull))
        if len(corners) == 18 and len(hull) ==11:
            print (len(corners), len(hull))

        #if len(corners) == 10 and len(hull) == 7:
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
            x = depthVal * (int(np.round(tmp[0][0]))-cam.cx + 0.5 ) / cam.fx
            y = depthVal * (int(np.round(tmp[0][1]))-cam.cy + 0.5 ) / cam.fy
            xpl = depthVal * (int(np.round(tmp[0][0]+4))-cam.cx + 0.5 ) / cam.fx,
            ypl = depthVal * (int(np.round(tmp[0][1]+4))-cam.cy + 0.5 ) / cam.fy
           # x-xpl, y-ypl
            errdepth = 0.04
            errx = errdepth * (int(np.round(tmp[0][0]))-cam.cx + 0.5 ) / cam.fx
            erry = errdepth * (int(np.round(tmp[0][1]))-cam.cy + 0.5 ) / cam.fy
            toterrx = sqrt(pow(errx,2)+ pow(x-xpl,2))
            toterry = sqrt(pow(erry,2)+ pow(y-ypl,2))

            print ("error X, T ", x, y,depthVal)
            #print ("Error on X, Y, Z ", toterrx, toterry, errdepth)
           # tmp_obj.append([x, y, depthVal] )
            tmp_obj.append([[x, y, depthVal], [toterrx, toterry, errdepth]])
        return tmp_obj
    except IndexError:
        print("No contour selected !!!")
        return tmp_obj

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="registrationMarker")
    parser.add_argument("-s", "--serialFile",dest="serialsFile", required=True,
                        help="Kinect Serials File", metavar="SERIALS")

    args = parser.parse_args()

  #  print sorted(vars(args).items())

    serialsListContourRot=[]
    # for key, value in sorted(vars(args).items()):
    #     if key is "serials":
    #         serialsList=value

    # print ("serials ", serialsList)

    with open(args.serialsFile, "r") as infile:
        for line in infile:
            serialsListContourRot.append((line.split()[0], line.split()[1]))

    print ("serials ", serialsListContourRot)
    referenceSerial=""
    for k, value in enumerate(serialsListContourRot):
	
        print ("serial", k)
        if (k==0):
            referenceSerial=value[0]
            continue
       # else:
       #      referenceSerial= serialsListContourRot[k-1][0]


        os.system("/home/hamit/libfreenect2pclgrabber/devel/lib/kinect2grabber/rosPairKinectv2Viewer  --serials "+ referenceSerial+" "+value[0])


        print ("reference serial", value[0])

        print("clockwise", bool(int(value[1])))


        contoursList=[]
        clockwise=True
        objpoints=[]


        for i, sr in enumerate([referenceSerial, value[0]]):
            imgFile=sr+"_"+str(i)+".png"
            if imgFile is None:
                continue
            depthFile=  imgFile.split('.')[0] + "_depth.png"
            exists = os.path.isfile(depthFile) and os.path.isfile(imgFile)
            if not exists:
                break
            img, imgray, depth = loadColorDepthImage(imgFile, depthFile)



            windowName="TrackbarGrayScale"
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName, 900,900)

            mask = np.zeros(img.shape, np.uint8)
            cv2.createTrackbar( 'GrayScale', windowName, 0, 255, nothing)



            contours_found=True
            if (i > 0 and bool(int(value[1]) == True)):
                clockwise= not clockwise
            isbreak=False

            while(1):

                imgcp=img.copy()

                grayScale = cv2.getTrackbarPos('GrayScale',windowName)

                contours = createContour(imgray, grayScale, clockwise )


                if len(contours) > 0:
                    cv2.drawContours(imgcp, contours, -1, (0, 255, 0), 1)
                    contours_found=True
                    for contour in contours[0]:
                        cv2.circle(imgcp, tuple(contour[0]), 1, (0, 0, 255) ,  -1);
                    cv2.imshow("TrackbarGrayScale", imgcp)
                elif len(contours) == 0 and contours_found :
                    cv2.drawContours(mask, contours,-1, (255,255,255,255), 1)

                    removed = cv2.add(imgcp, mask)
                    cv2.imshow("TrackbarGrayScale", removed)

                else:
                    cv2.imshow("TrackbarGrayScale", imgcp)




                imgcp=[]

                c=cv2.waitKey(2)

                if c == ord('s') :

                    tmpcamStr=imgFile.split('.')[0].split('/')[-1].split('_')[0]

                    cont=computeCoords(contours,depth, camera(tmpcamStr))

                    if len(contours) < 1:
                          continue
                    objpoints.append(cont)
                    contoursList.append([contours,  imgFile.split('.')[0].split('/')[-1]])

                    break
                elif c == ord('q') :
                    isbreak=True
                    break

            cv2.destroyAllWindows()

        if isbreak is True:
            continue
        print ("ContoursList \n", contoursList)
        objpointsNp=np.array(objpoints)
        A=np.asmatrix(objpointsNp[1][:,0])
        B=np.asmatrix(objpointsNp[0][:,0])
        Aerr=np.asmatrix(objpointsNp[1][:,1])
        Berr=np.asmatrix(objpointsNp[0][:,1])
        print('{0} and \n {1}'.format(Aerr, Berr))
        #printf("A,\n B")
        # A=np.asmatrix(objpoints[1][:,0])
        # B=np.asmatrix(objpoints[0][:,0])
        #Ifit()
        Ifit2()
        # par=[2]*16
        # fun(par)

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
        for increment,arg in enumerate([referenceSerial, value[0]]):

            a=arg.split('.')[0].split('/')[-1]
            transOutputFile+=a
            if increment  < 1:
                transOutputFile+="_to_"

            #print ("fileName: ",transOutputFile)

        np.savetxt(transOutputFile+'.out', trans , delimiter=' ',fmt='%1.8f')
        np.savetxt(str(k)+'.out', trans , delimiter=' ',fmt='%1.8f')
