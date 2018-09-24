import numpy as np
import cv2
import os
import glob
from scipy.optimize import minimize
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import math
import ROOT
from copy import deepcopy
from array import array

pointsCoordsIRandUVHD=np.array([])
coordsIR=np.array([])
uvsHD=np.array([])

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


   # R = np.dot(R_x, np.dot( R_y, R_z ))
    R = np.dot(R_z, np.dot( R_y, R_x ))
   # R= np.matrix(R_z)* np.matrix(R_y)* np.matrix(R_x)
    return R

def loadfromFiles(dirname):
    pointsVectorIR=[]
    pointsVectorHD=[]
    for file in sorted(os.listdir(dirname),key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])):
    #filelist=os.listdir(dirname)
    #for file in filelist.sort(key=lambda f: int(filter(str.isdigit, f))):
        if file.endswith(".xml"):
            print (dirname+file)
            store= cv2.FileStorage(dirname+file, cv2.FileStorage_READ)
            n1 = store.getNode("keypointsIR")
            n2 = store.getNode("keypointsHD")
            keypointvectorIR=[]
            keypointvectorHD=[]
            if n1.isSeq() and n2.isSeq():
                for el in range(0, n1.size()):
                    keypointvectorIR.append(n1.at(el).real())
                for el in range(0, n2.size()):
                    keypointvectorHD.append(n2.at(el).real())
            else :
                print ("keypoints are not sequence continue on next file")
                continue

            pointsVectorIR.append(keypointvectorIR)
            pointsVectorHD.append(keypointvectorHD)
    pointsVectorIR=np.array(pointsVectorIR)
    pointsVectorHD=np.array(pointsVectorHD)

    pointsVectorIR=pointsVectorIR.reshape(-1,3)
    pointsVectorHD=pointsVectorHD.reshape(-1,2)
    #print(pointsVectorIR)
    #print(pointsVectorHD)
    pointsVectorIRandHD=np.append(pointsVectorIR,pointsVectorHD ,axis=1)
    return (pointsVectorIRandHD)


def convertUVs2Coordinates(pointsUVsIRandHD):
    fx_ir=364.963
    fy_ir=364.963
    cx_ir=512./2
    cy_ir=424./2
    pointsCoordsIRandHD=[]
    for row in pointsUVsIRandHD:
        depth_val=row[2]/1000.
        if (depth_val >= 0.001):
            x_ir = (row[0] + 0.5 - cx_ir) * (1./fx_ir) * depth_val
            y_ir = (row[1] + 0.5 - cy_ir) * (1./fy_ir) * depth_val
            z_ir = depth_val
            pointsCoordsIRandHD.append([x_ir, y_ir, z_ir, row[3],row[4]])

    return (np.array(pointsCoordsIRandHD))

def transform(par, coordsIRx, coordsIRy, coordsIRz):
    #global pointsCoordsIRandUVHD

    #print (coordsIRx, coordsIRy, coordsIRz )

    rot= eulerAnglesToRotationMatrix(theta=[par[0], par[1], par[2]])
    #rot = np.matrix(euler2mat(par[0], par[1], par[2]))
    transl0=np.array([[par[3]],[par[4]],[par[5]]])
    trans=np.append(rot,transl0, axis=1)
    transl1=np.array([[0,0,0,1]])
    transfinal=np.append(trans,transl1, axis=0)
    pointIRcoords=np.array([[coordsIRx],
                            [coordsIRy],
                            [coordsIRz],
                            [1]
                           ])


    transformedCoordsIR=np.dot(transfinal,pointIRcoords)
    #print (transformedCoordsIR)

    #print (transformedCoordsIR)
    fx_hd=1081.37
    fy_hd=1081.37
    cx_hd=959.5
    cy_hd=539.5

    # x= par[6] *  row[0] /row[2]  + par[7]
    # y= par[6] *  row[1] /row[2]  + par[8]
    x = fx_hd * transformedCoordsIR[0][0] /transformedCoordsIR[2][0] + cx_hd
    y = fy_hd * transformedCoordsIR[1][0] /transformedCoordsIR[2][0] + cy_hd

    #print (x,y)


    return x,y



def ErrorFunc(par, coordsIRx, coordsIRy, coordsIRz, uvsHDx, uvsHDy):
    # uvsHD=np.array([[uvsHDx],
    #                 [uvsHDy]])
    # print (uvsHD)
    x, y =transform(par, coordsIRx, coordsIRy, coordsIRz)


    return pow(x-uvsHDx,2)+pow(y-uvsHDy,2)

def sum_sq(par, coordsIRx, coordsIRy, coordsIRz, uvsHDx, uvsHDy):
    """Sum of squares of the errors. This function is minimized"""
    return np.sum(ErrorFunc(par, coordsIRx, coordsIRy, coordsIRz, uvsHDx, uvsHDy)**2)

def min():
   # data provided
   #par=[0]*9;
   # fcnchi2(par)
   # coordsIR = pointsCoordsIRandUVHD[:,0:3]
   coordsIRx = pointsCoordsIRandUVHD[:,0:1].flatten()
   coordsIRy = pointsCoordsIRandUVHD[:,1:2].flatten()
   coordsIRz = pointsCoordsIRandUVHD[:,2:3].flatten()
   #print (type(coordsIRz))
   # uvsHD = pointsCoordsIRandUVHD[:,3:]
   uvsHDx = pointsCoordsIRandUVHD[:,3:4].flatten()
   uvsHDy = pointsCoordsIRandUVHD[:,4:5].flatten()
   # print (uvsHDy.flatten().shape[0])
   # x=np.array([1.0,2.5,3.5,4.0,1.1,1.8,2.2,3.7])
   # y=np.array([6.008,15.722,27.130,33.772,5.257,9.549,11.098,28.828])
   # here, create lambda functions for Line, Quadratic fit
   # tpl is a tuple that contains the parameters of the fit
   # funcLine=lambda tpl,x : tpl[0]*x+tpl[1]
   # funcQuad=lambda tpl,x : tpl[0]*x**2+tpl[1]*x+tpl[2]
   # func is going to be a placeholder for funcLine,funcQuad or whatever
   # function we would like to fit
   # func=funcLine
   # ErrorFunc is the diference between the func and the y "experimental" data
   #  ErrorFunc=lambda par,x,y: fcnchi2(par,x)-y
   #tplInitial contains the "first guess" of the parameters

   pStart  = (  0.01, 0.01, 0.01,
                0,  0.0, -0.0
              #  1081.37, 960., 540.
               # 364.963, 512./2, 424./2.
              )

   # leastsq finds the set of parameters in the tuple tpl that minimizes
   # ErrorFunc=yfit-yExperimental
   pfit, pcov, infodict, errmsg, success = leastsq(ErrorFunc, pStart[:], args=(coordsIRx, coordsIRy, coordsIRz, uvsHDx, uvsHDy),
                                                   full_output=1, epsfcn=0.0001)
   #pfit = minimize(sum_sq, pStart[:], args=(coordsIRx, coordsIRy, coordsIRz, uvsHDx, uvsHDy))

   print (" fit result: ",pfit, success)
   print (eulerAnglesToRotationMatrix(pfit))


def fcnchi2(npar, gin, f, par, iflag):
    rot= eulerAnglesToRotationMatrix(theta=[par[0], par[1], par[2]])
    #rot = np.matrix(euler2mat(par[0], par[1], par[2]))
    transl0=np.array([[par[3]],[par[4]],[par[5]]])
    trans=np.append(rot,transl0, axis=1)
    transl1=np.array([[0,0,0,1]])
    transfinal=np.append(trans,transl1, axis=0)
    # coordsIR=  pointsCoordsIRandUVHD[:,0:3]
    # coordsIR=np.transpose(np.append(coordsIR, np.ones((coordsIR.shape[0],1)), axis=1))
    transformedCoordsIR=np.transpose(np.dot(transfinal,coordsIR))

    uvsFound=[]
    for row in transformedCoordsIR:
        x= par[6] * row[0] / row[2]  + par[7]
        y= par[6] * row[1] / row[2]  + par[8]
        uvsFound.append([x,y])

    # uvsHD= pointsCoordsIRandUVHD[:,3:]

    deltaAandB = np.subtract(uvsFound,uvsHD)
    chi2SquareAandB=np.square(deltaAandB)

    f[0] = np.sum(chi2SquareAandB)




def Ifit():

    min=ROOT.TVirtualFitter.Fitter(0,9)
    min.SetFCN(fcnchi2)
    arglist = np.array( 10 * [0.])
    #arglist = array('d', 10 * [0.])
    ierflg = ROOT.Long(1982)

    arglist[0] = 1
    min.ExecuteCommand("SET PRINT",arglist,ierflg)
    vstart = np.array([0., 0., 0.,
                       0.,  0.,  0. ,
                       1081.37, 960., 540.
                       ])
    step = np.array([0.0001, 0.0001, 0.0001,
                     0.001, 0.001, 0.001,
            		 0.05000, 0.05000, 0.5000

                     ])
    min.SetParameter(0, "par0", vstart[0], step[0], -math.pi, math.pi)
    min.SetParameter(1, "par1", vstart[1], step[1], -math.pi, math.pi)
    min.SetParameter(2, "par2", vstart[2], step[2], -math.pi, math.pi)
    min.SetParameter(3, "par3", vstart[3], step[3], -2, 2)
    min.SetParameter(4, "par4", vstart[4], step[4], -2, 2)
    min.SetParameter(5, "par5", vstart[5], step[5], -2, 2)
    min.SetParameter(6, "par6", vstart[6], step[6],  1000, 1200)
    min.SetParameter(7, "par7", vstart[7], step[7],  850, 1000)
    min.SetParameter(8, "par8", vstart[8], step[8],  450, 600)
 	# min.SetParameter(9, "par9", vstart[9], step[9],  320, 380)
	# min.SetParameter(10, "par10", vstart[10], step[10],  220, 290)
	# min.SetParameter(11, "par11", vstart[11], step[11],  180, 250)

    min.FixParameter(6)
    min.FixParameter(7)
    min.FixParameter(8)
    # min.FixParameter(9)
    # min.FixParameter(10)
    # min.FixParameter(11)


    arglist[0] = ROOT.Long(50000)
    arglist[1] = ROOT.double(0.00001)
   # min.ExecuteCommand("MIGRAD", arglist, 2)
   # min.ExecuteCommand("HESSE", arglist, 2)
    min.ExecuteCommand("MINOS",arglist,2)

    #Print results
    amin, edm, errdef = ROOT.Double(0.18), ROOT.Double(0.19), ROOT.Double(0.20)
    nvpar, nparx, icstat = ROOT.Long(1983), ROOT.Long(1984), ROOT.Long(1985)
    min.GetStats(amin, edm, errdef, nvpar, nparx)
    min.PrintResults(3, amin)
   # print ("amin: ",amin)

    par =  ROOT.Double()
    error= ROOT.Double()
    parlist=[]
    for i in range(9):
        par0=min.GetParameter(i)
        #print (min.GetParError(i))
        parlist.append(deepcopy(par0))


    #print parlist
    print ('Rot Mat')
    ret_R = eulerAnglesToRotationMatrix(parlist[0:3])
    ret_t= np.transpose(np.matrix(parlist[3:6]))
    #print (np.matrix(ret_R), np.matrix(ret_t))
    print('Rot TvirturalFitter \n{0}  \n and translation   \n {1}'.format(np.matrix(ret_R), np.matrix(ret_t)))

    #print 'Translation'

    #print ret_t
    print ("")

    print (min.GetNumberFreeParameters (), min.GetNumberTotalParameters ())

    trans= np.vstack([np.hstack((ret_R,ret_t)),[0,0,0,1]])
   # np.savetxt("tranformation"+'.out', trans , delimiter=' ',fmt='%1.8f')

    matr=np.array(np.identity(4));
    matr[0,3] = 1.5
    matr[1,3] = 1.5
    matr[2,3] = -0.3

    trans = np.dot(matr,np.linalg.inv(trans))
    print (trans)
    with open("tranformation"+".out", 'w') as f:
        a=np.array(["Tvector"])
        np.savetxt(f, a, delimiter=" ",fmt="%s")
        np.savetxt(f, trans[0:3,3] , delimiter=' ', fmt='%1.8f')
        a=np.array(["\nRMatrix"])
        np.savetxt(f, a, delimiter=" ",fmt="%s")
        np.savetxt(f, trans[0:3,0:3], delimiter=' ',fmt='%1.8f',)
        a=np.array(["\nCamera Intrinsics: focal height width"])
        np.savetxt(f, a, delimiter=" ",fmt="%s")
        np.savetxt(f, np.atleast_2d([parlist[6], parlist[8]*2, parlist[7]*2]) , delimiter=' ',fmt='%1.8g')

def Ifit2():
    min=ROOT.TMinuit(9)
    min.SetFCN(fcnchi2)
    arglist = np.array( 10 * [0.])
    ierflg = ROOT.Long(1982)

    arglist[0] = 1

    min.mnexcm("SET ERR", arglist, 1, ierflg)
    vstart = np.array([0., 0., 0.,
                       0.,  0.,  0. ,
                       1081.37, 960., 540.
                       ])
    step = np.array([0.0001, 0.0001, 0.0001,
                     0.001, 0.001, 0.001,
            		 0.05000, 0.05000, 0.5000

                     ])
    min.mnparm(0, "par0", vstart[0], step[0], -math.pi, math.pi, ierflg)
    min.mnparm(1, "par1", vstart[1], step[1], -math.pi, math.pi, ierflg)
    min.mnparm(2, "par2", vstart[2], step[2], -math.pi, math.pi, ierflg)
    min.mnparm(3, "par3", vstart[3], step[3], -2, 2, ierflg)
    min.mnparm(4, "par4", vstart[4], step[4], -2, 2, ierflg)
    min.mnparm(5, "par5", vstart[5], step[5], -2, 2, ierflg)
    min.mnparm(6, "par6", vstart[6], step[6],  1000, 1200, ierflg)
    min.mnparm(7, "par7", vstart[7], step[7],  850, 1000, ierflg)
    min.mnparm(8, "par8", vstart[8], step[8],  450, 600, ierflg)
 	# min.mnparm(9, "par9", vstart[9], step[9],  320, 380)
	# min.mnparm(10, "par10", vstart[10], step[10],  220, 290)
	# min.mnparm(11, "par11", vstart[11], step[11],  180, 250)

    min.FixParameter(6)
    min.FixParameter(7)
    min.FixParameter(8)
    # min.FixParameter(9)
    # min.FixParameter(10)
    # min.FixParameter(11)


    arglist[0] = ROOT.Long(5000000)
    arglist[1] = ROOT.double(0.001)
    min.mnexcm("MIGRAD", arglist, 2, ierflg)
    min.mnexcm("HESSE", arglist, 2,ierflg)
    #min.mnexcm("MINOS", arglist, 2, ierflg)

    #Print results
    amin, edm, errdef = ROOT.Double(0.18), ROOT.Double(0.19), ROOT.Double(0.20)
    nvpar, nparx, icstat = ROOT.Long(1983), ROOT.Long(1984), ROOT.Long(1985)
    min.mnstat(amin, edm, errdef, nvpar, nparx,icstat)
    min.mnprin(3, amin)
    print ("amin: ",amin)

    par =  ROOT.Double()
    error= ROOT.Double()


    parlist=[]
    for i in range(9):
        min.GetParameter(i, par, error)
        parlist.append(deepcopy(par))


    #print parlist
    print ('Rot Mat')
    ret_R = eulerAnglesToRotationMatrix(parlist[0:3])
    ret_t= np.transpose(np.matrix(parlist[3:6]))
    #print (np.matrix(ret_R), np.matrix(ret_t))
    print('Rot TvirturalFitter \n{0}  \n and translation   \n {1}'.format(np.matrix(ret_R), np.matrix(ret_t)))

    #print 'Translation'

    #print ret_t
    print ("")

   # print (min.GetNumberFreeParameters (), min.GetNumberTotalParameters ())

    trans= np.vstack([np.hstack((ret_R,ret_t)),[0,0,0,1]])
   # np.savetxt("tranformation"+'.out', trans , delimiter=' ',fmt='%1.8f')

    matr=np.array(np.identity(4))
    matr[0,3] = 1.5
    matr[1,3] = 1.5
    matr[2,3] = -0.3

    trans = np.dot(matr,np.linalg.inv(trans))
    print (trans)
    with open("tranformation"+".out", 'w') as f:
        a=np.array(["Tvector"])
        np.savetxt(f, a, delimiter=" ",fmt="%s")
        np.savetxt(f, trans[0:3,3] , delimiter=' ', fmt='%1.8f')
        a=np.array(["\nRMatrix"])
        np.savetxt(f, a, delimiter=" ",fmt="%s")
        np.savetxt(f, trans[0:3,0:3], delimiter=' ',fmt='%1.8f',)
        a=np.array(["\nCamera Intrinsics: focal height width"])
        np.savetxt(f, a, delimiter=" ",fmt="%s")
        np.savetxt(f, np.atleast_2d([parlist[6], parlist[8]*2, parlist[7]*2]) , delimiter=' ',fmt='%1.8g')


if __name__ == '__main__':

    pointsUVsIRandHD = loadfromFiles('/home/hamit/calibrateKinectv2/bin/test3/')
    pointsCoordsIRandUVHD = convertUVs2Coordinates(pointsUVsIRandHD)
    uvsHD= pointsCoordsIRandUVHD[:,3:]
    coordsIR=  pointsCoordsIRandUVHD[:,0:3]
    coordsIR=np.transpose(np.append(coordsIR, np.ones((coordsIR.shape[0],1)), axis=1))

    # coordsIR = pointsCoordsIRandUVHD[:,0:3]
    # coordsIR=  np.append(coordsIR, np.ones((coordsIR.shape[0],1)), axis=1)
    #print (coordsIR.T)
    min()
    Ifit()
