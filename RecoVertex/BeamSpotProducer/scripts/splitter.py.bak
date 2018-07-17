#!/usr/bin/env python
import sys,os,commands
from CommonMethods import *

class FileObj:
    def __init__(self):
        self.run       = 0
        self.size      = 0
        self.fileNames = []                 

def getRunNumberFromFileName(fileName):
    regExp = re.search('(\D+)_(\d+)_(\d+)_(\d+)',fileName)
    if not regExp:
        return -1
    return long(regExp.group(3))
                


def main():
    if len(sys.argv) < 2:
        error = "Usage: splitter fromDir"
        exit(error)
    sourceDir = sys.argv[1] + '/'

    fileList = ls(sourceDir,".txt")

    fileObjList = {}

    totalSize = 0
    for fileName in fileList:
        runNumber = getRunNumberFromFileName(fileName)
        if runNumber not in fileObjList:
            fileObjList[runNumber] = FileObj()
            fileObjList[runNumber].run = runNumber 
        fileObjList[runNumber].fileNames.append(fileName) 
        aCommand  = 'ls -l '+ sourceDir + fileName 
        output = commands.getstatusoutput( aCommand )
        fileObjList[runNumber].size += int(output[1].split(' ')[4])
        totalSize += int(output[1].split(' ')[4]) 

    sortedKeys = sorted(fileObjList.keys())

    split=13

    dirSize = 0
    tmpList = []
    for run in sortedKeys:
        dirSize += fileObjList[run].size
        tmpList.append(fileObjList[run])
        if dirSize > totalSize/split or run == sortedKeys[len(sortedKeys)-1]:
            newDir = sourceDir + "Run" + str(tmpList[0].run) + "_" + str(tmpList[len(tmpList)-1].run) + "/"
            aCommand  = 'mkdir '+ newDir
            output = commands.getstatusoutput( aCommand )
            print str(100.*dirSize/totalSize) + "% " + "Run" + str(tmpList[0].run) + "_" + str(tmpList[len(tmpList)-1].run) 
            for runs in tmpList:
                #print 'cp '+ sourceDir + runs.fileNames[0] + " " + newDir
                cp(sourceDir,newDir,runs.fileNames) 
            tmpList = []
            dirSize = 0
        


    
    print totalSize
    print sortedKeys 
    exit("ok")    






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
