#!/usr/bin/env python
import sys,os,commands
from CommonMethods import *

def main():
    if len(sys.argv) < 3:
        error = "Usage: copyFiles.py fromDir destDir (optional filter)"
        exit(error)
    fromDir = sys.argv[1]
    if (fromDir[len(fromDir)-1] != '/'):
        fromDir += '/'
    destDir = sys.argv[2] + "/"
    filter = ""
    if len(sys.argv) > 3:
        filter = sys.argv[3]
    fileList = ls(fromDir,filter)
    copiedFiles = cp(fromDir,destDir,fileList)

    if len(copiedFiles) != len(fileList):
        error = "ERROR: I couldn't copy all files from " + fromDir
        exit(error)
        
if __name__ == "__main__":
    main()
