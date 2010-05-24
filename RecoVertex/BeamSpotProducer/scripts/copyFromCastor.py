#!/usr/bin/env python
import sys,os,commands
from CommonMethods import *
def main():
    if len(sys.argv) < 3:
        print "Usage: cpFromCastor fromDir toDir (optional runnumber)"
        exit(0)
    user = os.getenv("USER")
    castorDir = "/castor/cern.ch/cms/store/caf/user/" + user + "/" + sys.argv[1] + "/"
    aCommand = "nsls " + castorDir
    
    if len(sys.argv) > 3:
        aCommand += " | grep " + sys.argv[3]
    output = commands.getstatusoutput(aCommand)
    if output[0] != 0:
        print output[1]
        exit(0)
    fileList = output[1].split('\n')
    destDir = sys.argv[2]
    copiedFiles = cp(castorDir,destDir,fileList)

    if len(copiedFiles) != len(fileList):
        error = "ERROR: I couldn't copy all files from castor"
        exit(error)
        
if __name__ == "__main__":
    main()
