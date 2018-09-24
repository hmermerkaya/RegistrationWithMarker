#!/home/hamit/anaconda2/bin/python
import numpy as np
import cv2
from math import sqrt


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="registrationMarker")
    parser.add_argument("-f1", "--files1", dest="fileNames1", required=True,
                        help="1st file ", metavar="FILE")
    args = parser.parse_args()