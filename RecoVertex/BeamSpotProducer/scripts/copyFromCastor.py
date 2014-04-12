#!/usr/bin/env python
import sys,os,commands
from CommonMethods import *
def main():
    if len(sys.argv) < 3:
        error = "Usage: cpFromCastor fromDir toDir (optional filter)"
        exit(error)
    user = os.getenv("USER")
    castorDir = "/castor/cern.ch/cms/store/caf/user/" + user + "/" + sys.argv[1] + "/"
    filter = ""
    if len(sys.argv) > 3:
        filter = sys.argv[3]
    fileList = ls(castorDir,filter)
    destDir = sys.argv[2]
    copiedFiles = cp(castorDir,destDir,fileList)

    if len(copiedFiles) != len(fileList):
        error = "ERROR: I couldn't copy all files from castor"
        exit(error)
        
if __name__ == "__main__":
    main()
