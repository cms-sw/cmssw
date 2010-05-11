#!/usr/bin/env python
import sys,os,commands
from CommonMethods import *
def main():
    if len(sys.argv) < 3:
        error = "Usage: copyAndRename fromDir toDir"
        exit(error)
    sourceDir = sys.argv[1] + '/'
    destDir   = sys.argv[2] + '/'

    fileList = ls(sourceDir)
    if not os.path.isdir(destDir):
        error = "WARNING: destination directory doesn't exist! Creating it..."
        print error
        os.mkdir(destDir)
    copiedFiles = cp(sourceDir,destDir,fileList)

    if len(copiedFiles) != len(fileList):
        error = "ERROR: I couldn't copy all files from castor"
        exit(error)

    for fileName in fileList:
        fullFileName = destDir + fileName
        runNumber = -1;
        with open(fullFileName,'r') as file:
            for line in file:
                if line.find("Runnumber") != -1:
                    tmpRun = int(line.split(' ')[1])
                    if runNumber != -1 and tmpRun != runNumber:
                        error = "This file (" + fileName + ") contains more than 1 run number! I don't know how to deal with it!"
                        exit(error)
                    runNumber = int(line.split(' ')[1])
        file.close()
        newFileName = fileName.replace("None",str(runNumber))
        if fileName != newFileName:
            aCmd = "mv " + destDir + fileName + " " + destDir + newFileName
            print aCmd
            output =  commands.getstatusoutput(aCmd)
            if output[0] != 0:
                print output[1]
        else:
            print "WARNING couldn't find keyword None in file " + fileName




        
if __name__ == "__main__":
    main()
