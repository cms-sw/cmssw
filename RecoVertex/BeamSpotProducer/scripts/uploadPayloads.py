#!/usr/bin/env python
import sys,os,commands
from CommonMethods import *

class FileObj:
    def __init__(self):
        self.run      = 0
        self.iovSince = 0
        self.fileName = ''                 

                                                                        

def main():
    payloadDir = "./Payloads_Repro2010Nov09/"
    aCommand  = "ls " + payloadDir + " | grep BeamSpotObjects_2009_LumiBased_ | grep txt"           
    output = commands.getstatusoutput( aCommand )
    listOfFiles = output[1].split('\n')                                                                              
#    print listOfFiles
    dropbox = "/DropBox"
    for fileName in listOfFiles:
        fileNameRoot = fileName[0:len(fileName)-4]
        print fileNameRoot
        uploadSqliteFile(payloadDir, fileNameRoot, dropbox)
            


        
if __name__ == "__main__":
    main()
