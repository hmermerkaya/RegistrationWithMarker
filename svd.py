#from numpy import *
from math import sqrt
import numpy as np
#import opencvchessboard
import findContours

# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

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

  #  print  t

    return R, t




A=np.asmatrix(findContours.objpoints[1])
B=np.asmatrix(findContours.objpoints[0])

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

trans= (np.vstack([np.hstack((ret_R,ret_t)),[0,0,0,1]]))
np.savetxt('test.out',trans , delimiter=' ',fmt='%1.8f')

# print ("RMSE:", rmse)
# print ("If RMSE is near zero, the function is correct!")