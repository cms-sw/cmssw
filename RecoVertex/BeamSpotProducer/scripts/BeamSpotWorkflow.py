#!/usr/bin/env python
#____________________________________________________________
#
#  BeamSpotWorkflow
#
# A very complicate way to automatize the beam spot workflow
#
# Francisco Yumiceva, Lorenzo Uplegger
# yumiceva@fnal.gov, uplegger@fnal.gov
#
# Fermilab, 2010
#
#____________________________________________________________

"""
   BeamSpotWorkflow.py

   A very complicate script to upload the results into the DB

   usage: %prog -d <data file/directory> -t <tag name>
   -o, --overwrite : Overwrite results files when copying.
   -T, --Test      : Upload files to Test dropbox for data validation.   
   -u, --upload    : Upload files to offline drop box via scp.
   -z, --zlarge    : Enlarge sigmaZ to 10 +/- 0.005 cm.

   Francisco Yumiceva (yumiceva@fnal.gov)
   Lorenzo Uplegger   (send an email to Francisco)
   Fermilab 2010
   
"""


import sys,os
import commands, re, time
import datetime
import ConfigParser
import optparse
from BeamSpotObj import BeamSpot
from IOVObj import IOV
from CommonMethods import *

#####################################################################################
# lumi tools CondCore/Utilities/python/timeUnitHelper.py
#####################################################################################
def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

########################################################################
def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)

########################################################################
def unpackLumiid(i):
    """unpack 64bit lumiid to dictionary {'run','lumisection'}
    """
    j=unpack(i)
    return {'run':j[0],'lumisection':j[1]}

#####################################################################################
# General functions
#####################################################################################
def getLastUploadedIOV(tagName,destDB="oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT"):
    listIOVCommand = "cmscond_list_iov -c " + destDB + " -P /afs/cern.ch/cms/DB/conddb -t " + tagName 
    aCommand       = listIOVCommand + " | grep DB= | tail -1 | awk \'{print $1}\'"
#    print " >> " + aCommand
    output = commands.getstatusoutput( aCommand )
    if output[1] == '' :
        dbError = commands.getstatusoutput( listIOVCommand )
        exit("ERROR: Can\'t connect to db because:\n" + dbError[1])
    #WARNING when we pass to lumi IOV this should be long long
    return long(output[1])

########################################################################
def getListOfFilesToProcess(dataSet,lastRun=-1):
    queryCommand = "dbs --search --query \"find file where dataset=" + dataSet
    if lastRun != -1:
        queryCommand = queryCommand + " and run > " + str(lastRun)
    queryCommand = queryCommand + "\" | grep .root"    
#    print " >> " + queryCommand
    output = commands.getstatusoutput( queryCommand )
    return output[1].split('\n')

########################################################################
def getLastClosedRun(DBSListOfFiles):
    runs = []
    for file in DBSListOfFiles:
        runNumber = getRunNumberFromDBSName(file)
        if runs.count(runNumber) == 0: 
            runs.append(runNumber)

    if len(runs) <= 1: #No closed run
        return -1
    else:
        runs.sort()
        return long(runs[len(runs)-2])
    
########################################################################
def ls(dir):
    lsCommand      = ''
    listOfFiles    = []
    if dir.find('castor') != -1:
    	lsCommand = 'ns'
    elif not os.path.exists(dir):
        print "ERROR: File or directory " + dir + " doesn't exist"
        return listOfFiles

    aCommand  = lsCommand  + 'ls '+ dir + " | grep .txt"

    tmpStatus = commands.getstatusoutput( aCommand )
    listOfFiles = tmpStatus[1].split('\n')
    if len(listOfFiles) == 1:
        if listOfFiles[0].find('No such file or directory') != -1:
            exit("ERROR: File or directory " + path + " doesn't exist") 

    return listOfFiles            

########################################################################
def dirExists(dir):
    if dir.find("castor") != -1:
    	lsCommand = "nsls " + dir
        output = commands.getstatusoutput( lsCommand )
        return not output[0]
    else:
        return os.path.exists(dir)

########################################################################
def getRunNumberFromFileName(fileName):
    regExp = re.search('(\w+)_(\d+)_(\d+)_(\d+)',fileName)
    if not regExp:
        return -1
    return long(regExp.group(3))

########################################################################
def getRunNumberFromDBSName(fileName):
    regExp = re.search('(\w+)/(\d+)/(\d+)/(\d+)/(\w+)',fileName)
    if not regExp:
        return -1
    return long(regExp.group(3)+regExp.group(4))
    
########################################################################
def getNewRunList(fromDir,lastUploadedIOV):
    newRunList = []
    listOfFiles = ls(fromDir)
    runFileMap = {}
    for fileName in listOfFiles:
        runNumber = getRunNumberFromFileName(fileName) 
        if runNumber > lastUploadedIOV:
            newRunList.append(fileName)
    return newRunList        

########################################################################
def selectFilesToProcess(listOfFilesToProcess,newRunList):
    selectedFiles = []
    runsToProcess = {}
    processedRuns = {}
    for file in listOfFilesToProcess:
        run = getRunNumberFromDBSName(file)
#        print "To process: " + str(run) 
        if run not in runsToProcess:
            runsToProcess[run] = 1
        else:
            runsToProcess[run] = runsToProcess[run] + 1 

    for file in newRunList:
        run = getRunNumberFromFileName(file)
#        print "Processed: " + str(run)
        if run not in processedRuns:
            processedRuns[run] = 1
        else:
            processedRuns[run] = processedRuns[run] + 1 

    #WARNING: getLastClosedRun MUST also have a timeout otherwise the last run will not be considered
    lastClosedRun = getLastClosedRun(listOfFilesToProcess)
#    print "LastClosedRun:-" + str(lastClosedRun) + "-"

    processedRunsKeys = processedRuns.keys()
    processedRunsKeys.sort()

    for run in processedRunsKeys:
        if run <= lastClosedRun :
            print "For run " + str(run) + " I have processed " + str(processedRuns[run]) + " files and in DBS there are " + str(runsToProcess[run]) + " files!"
            if processedRuns[run] == runsToProcess[run]:
                for file in newRunList:
                    if run == getRunNumberFromFileName(file):
                        selectedFiles.append(file)
            else:
                exit("ERROR: For run " + str(run) + " I have processed " + str(processedRuns[run]) + " files but in DBS there are " + str(runsToProcess[run]) + " files!")
    return selectedFiles            

########################################################################
def cp(fromDir,toDir,listOfFiles):
    cpCommand   = ''
    copiedFiles = []
    if fromDir.find('castor') != -1:
    	cpCommand = 'rf'

    for file in listOfFiles:
        if os.path.isfile(toDir+file):
            if option.overwrite:
                print "File " + file + " already exists in destination directory. We will overwrite it."
            else:
                print "File " + file + " already exists in destination directory. We will Keep original file."
                copiedFiles.append(file)
                continue
    	# copy to local disk
    	aCommand = cpCommand + 'cp '+ fromDir + file + " " + toDir
    	print " >> " + aCommand
        tmpStatus = commands.getstatusoutput( aCommand )
        if tmpStatus[0] == 0:
            copiedFiles.append(file)
        else:
            print "[cp()]\tERROR: Can't copy file " + file
    return copiedFiles

########################################################################
def sendEmail(mailList,error):
    print "Sending email to " + mailList + " with body: " + error


########################################################################
if __name__ == '__main__':

    ######### COMMAND LINE OPTIONS ##############
    option,args = parse(__doc__)

    #Right now always in the test DB
#    destDB = 'oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT'
    destDB = 'oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT'
    if option.Test:
        destDB = 'oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT'

    ######### CONFIGURATION FILE ################
    configurationFile = 'BeamSpotWorkflow.cfg'
    configuration     = ConfigParser.ConfigParser()
    print 'Reading configuration from ', configurationFile
    configuration.read(configurationFile)

    sourceDir   = configuration.get('Common','SOURCE_DIR')
    archiveDir  = configuration.get('Common','ARCHIVE_DIR')
    workingDir  = configuration.get('Common','WORKING_DIR')
    databaseTag = configuration.get('Common','DBTAG')
    dataSet     = configuration.get('Common','DATASET')
    IOVBase     = configuration.get('Common','IOV_BASE')
    mailList    = configuration.get('Common','EMAIL')

    ######### DIRECTORIES SETUP #################
    if sourceDir[len(sourceDir)-1] != '/':
        sourceDir = sourceDir + '/'
    if not dirExists(sourceDir):
        error = "ERROR: The source directory " + sourceDir + " doesn't exist!"
        sendEmail(mailList,error)
        exit(error)

    if archiveDir[len(archiveDir)-1] != '/':
        archiveDir = archiveDir + '/'
    if not os.path.isdir(archiveDir):
	os.mkdir(archiveDir)

    if workingDir[len(workingDir)-1] != '/':
        workingDir = workingDir + '/'
    if not os.path.isdir(workingDir):
	os.mkdir(workingDir)
    else:
        os.system("rm -f "+ workingDir + "*") 


#    lastUploadedIOV = getLastUploadedIOV(databaseTag,destDB) 
    lastUploadedIOV = 133885

    ######### Get list of files processed after the last IOV  
    newProcessedRunList      = getNewRunList(sourceDir,lastUploadedIOV)
    if len(newProcessedRunList) == 0:
        exit("There are no new runs after " + str(lastUploadedIOV))

    ######### Copy files to archive directory
    for i in range(3):
        copiedFiles = cp(sourceDir,archiveDir,newProcessedRunList)    
        if len(copiedFiles) == len(newProcessedRunList):
            break;
    if len(copiedFiles) != len(newProcessedRunList):
        error = "ERROR: I can't copy more than " + str(len(copiedFiles)) + " files out of " + str(len(newProcessedRunList)) 
        sendEmail(mailList,error)
        exit(error)


    ######### Get from DBS the list of files after last IOV    
    listOfFilesToProcess = getListOfFilesToProcess(dataSet,lastUploadedIOV) 

    ######### Get list of files to process for DB
    selectedFilesToProcess = selectFilesToProcess(listOfFilesToProcess,copiedFiles)

    ######### Copy files to working directory
    for i in range(3):
        copiedFiles = cp(archiveDir,workingDir,selectedFilesToProcess)    
        if len(copiedFiles) == len(selectedFilesToProcess):
            break;
    if len(copiedFiles) != len(selectedFilesToProcess):
        error = "ERROR: I can't copy more than " + str(len(copiedFiles)) + " files out of " + str(len(selectedFilesToProcess)) 
        sendEmail(mailList,error)
        exit(error)

    beamSpotObjList = []
    for fileName in copiedFiles:
        readBeamSpotFile(workingDir+fileName,beamSpotObjList,IOVBase)

    sortAndCleanBeamList(beamSpotObjList,IOVBase)

    payloadFileName = "PayloadFile.txt"

    payloadList = createWeightedPayloads(workingDir+payloadFileName,beamSpotObjList,True)

    tmpPayloadFileName = workingDir + "SingleTmpPayloadFile.txt"
    tmpSqliteFileName  = workingDir + "SingleTmpSqliteFile.txt"
    ##############################################################
    #WARNING timetype is fixed to run
    timetype = 'runnumber'
    ##############################################################
    writeDBTemplate = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/write2DB_template.py"
    readDBTemplate  = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/readDB_template.py"
    for payload in payloadList:
        if option.zlarge:
            payload.sigmaZ = 10
            payload.sigmaZerr = 2.5e-05
        tmpFile = file(tmpPayloadFileName,'w')
        dumpValues(payload,tmpFile)
        tmpFile.close()
        if not writeSqliteFile(tmpSqliteFileName,databaseTag,timetype,tmpPayloadFileName,writeDBTemplate,workingDir):
            print "An error occurred while writing the sqlite file: " + tmpSqliteFileName

        readSqliteFile(tmpSqliteFileName,databaseTag,readDBTemplate,workingDir)

        
        ##############################################################
        #WARNING iovs are fixed on runumber
        iov_since = payload.Run
        iov_till  = iov_since
        appendSqliteFile("Combined.db", tmpSqliteFileName, databaseTag, iov_since, iov_till ,workingDir)

	os.system("rm -f " + tmpPayloadFileName)
        
        
   #Create and upload payloads
#    aCommand = "./createPayload.py -d " + workingDir+payloadFileName + " -t " + databaseTag
#    tmpStatus = commands.getstatusoutput( aCommand )
#    print aCommand
#    if tmpStatus[0] == 0:
#        print "Done!"
#    else:
#        print "Something wrong"
