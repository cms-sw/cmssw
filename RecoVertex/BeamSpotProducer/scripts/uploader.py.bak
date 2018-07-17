#!/usr/bin/env python
import sys,os,commands
from CommonMethods import *

class FileObj:
    def __init__(self):
        self.run      = 0
        self.iovSince = 0
        self.fileName = ''                 

                                                                        

def main():
    payloadDir = "./archive_repro_13May/payloads/"
    aCommand  = "ls " + payloadDir + " | grep BeamSpotObjects_2009_LumiBased_ | grep txt"           
    output = commands.getstatusoutput( aCommand )
    listOfFiles = output[1].split('\n')                                                                              
    print listOfFiles
    finalList = {}
    for fileName in listOfFiles:
        file = open(payloadDir + fileName)
        for line in file:
            if line.find("since") != -1:
                tmpObj = FileObj()
                tmpObj.run = unpackLumiid(long(line.split(' ')[1]))["run"]
                tmpObj.iovSince = line.split(' ')[1].replace('\n','')
                tmpObj.fileName = fileName
                finalList[tmpObj.run] = tmpObj
                file.close()
                break

    sortedKeys = sorted(finalList.keys())

    databaseTag = ''
    regExp = re.search('(\D+)(\d+)_(\d+)_(\w+)',listOfFiles[0])
    if regExp:
        databaseTag = regExp.group(4)
    else:
        exit("Can't find reg exp")

    uuid = commands.getstatusoutput('uuidgen -t')[1]
    final_sqlite_file_name = databaseTag + '@' + uuid
    megaNumber = "18446744073709551615"
    print final_sqlite_file_name
    for run in sortedKeys:
        appendSqliteFile(final_sqlite_file_name + ".db", payloadDir+finalList[run].fileName.replace(".txt",".db"), databaseTag, finalList[run].iovSince, megaNumber,payloadDir)
        print finalList[run].fileName.replace(".txt",".db")
    aCommand  = "cp " + payloadDir + finalList[sortedKeys[0]].fileName + " " + payloadDir + final_sqlite_file_name + ".txt"
    output = commands.getstatusoutput( aCommand )
    dropbox = "/DropBox"
    print sortedKeys[0]
    print finalList[sortedKeys[0]].fileName
#    uploadSqliteFile(payloadDir, final_sqlite_file_name, dropbox)
            


        
if __name__ == "__main__":
    main()
