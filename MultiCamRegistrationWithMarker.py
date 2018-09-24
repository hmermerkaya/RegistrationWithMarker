import argparse
import os

parser = argparse.ArgumentParser(description="registrationMarker")

parser.add_argument("-f", "--files", dest="fileNames", required=True, nargs=4,
                        help="1st file ", metavar="FILE")

# vars(args)
args = parser.parse_args()
dictArgs=vars(args)
print (dictArgs["fileNames"])
fileNames=dictArgs["fileNames"]
cmd="./RegistrationWithMarker.py -f1 "+ fileNames[0] + " -f2 " + fileNames[1]
os.system(cmd)
cmd="./RegistrationWithMarker.py -f1 "+ fileNames[2] + " -f2 " + fileNames[3]
os.system(cmd)
