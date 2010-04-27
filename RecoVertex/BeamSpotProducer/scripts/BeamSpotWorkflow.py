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
   -u, --upload : Upload files to offline drop box via scp.
   -z, --zlarge : Enlarge sigmaZ to 10 +/- 0.005 cm.
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Lorezo Uplegger    (send an email to Francisco)
   Fermilab 2010
   
"""


import sys,os
import commands, re, time
import datetime
import ConfigParser
import optparse

USAGE = re.compile(r'(?s)\s*usage: (.*?)(\n[ \t]*\n|$)')

def nonzero(self): # will become the nonzero method of optparse.Values
    "True if options were given"
    for v in self.__dict__.itervalues():
        if v is not None: return True
    return False

optparse.Values.__nonzero__ = nonzero # dynamically fix optparse.Values

class ParsingError(Exception): pass


def exit(msg=""):
    raise SystemExit(msg or optionstring.replace("%prog",sys.argv[0]))

def parse(docstring, arglist=None):
    global optionstring
    global tagType
    optionstring = docstring
    match = USAGE.search(optionstring)
    if not match: raise ParsingError("Cannot find the option string")
    optlines = match.group(1).splitlines()
    try:
        p = optparse.OptionParser(optlines[0])
        for line in optlines[1:]:
            opt, help=line.split(':')[:2]
            short,long=opt.split(',')[:2]
            if '=' in opt:
                action='store'
                long=long.split('=')[0]
            else:
                action='store_true'
            p.add_option(short.strip(),long.strip(),
                         action = action, help = help.strip())
    except (IndexError,ValueError):
        raise ParsingError("Cannot parse the option string correctly")
    return p.parse_args(arglist)

#####################################################################################
# lumi tools CondCore/Utilities/python/timeUnitHelper.py
#####################################################################################
def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)

def unpackLumiid(i):
    """unpack 64bit lumiid to dictionary {'run','lumisection'}
    """
    j=unpack(i)
    return {'run':j[0],'lumisection':j[1]}

#####################################################################################
# General functions
#####################################################################################

def getLastUploadedIOV(tagName):
    listIOVCommand = "cmscond_list_iov -c oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT -P /afs/cern.ch/cms/DB/conddb -t " + tagName 
    aCommand       = listIOVCommand + " | grep DB= | tail -1 | awk \'{print $1}\'"
#    print " >> " + aCommand
    output = commands.getstatusoutput( aCommand )
    if output[1] == '' :
        dbError = commands.getstatusoutput( listIOVCommand )
        exit("ERROR: Can\'t connect to db because:\n" + dbError[1])
    #WARNING when we pass to lumi IOV this should be long long
    return long(output[1])

def getListOfFilesToProcess(dataSet,lastRun=-1):
    queryCommand = "dbs --search --query \"find file where dataset=" + dataSet
    if lastRun != -1:
        queryCommand = queryCommand + " and run > " + str(lastRun)
    queryCommand = queryCommand + "\" | grep .root"    
#    print " >> " + queryCommand
    output = commands.getstatusoutput( queryCommand )
    return output[1].split('\n')

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
    
def ls(dir):
    lsCommand      = ''
    listOfFiles    = []
    if dir.find('castor') != -1:
    	lsCommand = 'ns'
    elif not os.path.exists(dir):
        exit("ERROR: File or directory " + dir + " doesn't exist") 

    aCommand  = lsCommand  + 'ls '+ dir + " | grep .txt"

    tmpStatus = commands.getstatusoutput( aCommand )
    listOfFiles = tmpStatus[1].split('\n')
    if len(listOfFiles) == 1:
        if listOffiles[0].find('No such file or directory') != -1:
            exit("ERROR: File or directory " + path + " doesn't exist") 

    return listOfFiles            

def getRunNumberFromFileName(fileName):
    regExp = re.search('(\w+)_(\d+)_(\d+)_(\d+)',fileName)
    if not regExp:
        return -1
    return long(regExp.group(3))

def getRunNumberFromDBSName(fileName):
    regExp = re.search('(\w+)/(\d+)/(\d+)/(\d+)/(\w+)',fileName)
    if not regExp:
        return -1
    return long(regExp.group(3)+regExp.group(4))
    

def getNewRunList(fromDir,lastUploadedIOV):
    newRunList = []
    listOfFiles = ls(fromDir)
    runFileMap = {}
    for fileName in listOfFiles:
        runNumber = getRunNumberFromFileName(fileName) 
        if runNumber > lastUploadedIOV:
            newRunList.append(fileName)
    return newRunList        

def getListOfFilesToCopy(listOfFilesToProcess,newRunList):
    listOfFilesToCopy = []
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
            if processedRuns[run] == runsToProcess[run]:
                for file in newRunList:
                    if run == getRunNumberFromFileName(file):
                        listOfFilesToCopy.append(file)
            else:
                exit("ERROR: I still have less files than the ones that are in DBS!")
    return listOfFilesToCopy            

def cp(fromDir,toDir,listOfFiles):
    cpCommand   = ''
    copiedFiles = []
    if fromDir.find('castor') != -1:
    	cpCommand = 'rf'
    elif not os.path.exists(fromDir):
        exit("ERROR: File or directory " + fromDir + " doesn't exist") 

    for file in listOfFiles:
        if os.path.isfile(toDir+file):
            if option.overwrite:
                print "File " + file + " already exists in destination. We will overwrite it."
            else:
                print "File " + file + " already exists in destination. Keep original file."
                copiedFiles.append(file)
                continue
    	    # copy to local disk
    	aCommand = cpCommand + 'cp '+ fromDir + file + " " + toDir
#    	print " >> " + aCommand
        tmpStatus = commands.getstatusoutput( aCommand )
        if tmpStatus[0] == 0:
            copiedFiles.append(file)
        else:
            exit("ERROR: Can't copy file " + file)
    return copiedFiles



if __name__ == '__main__':
    # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__)
    #    if not args and not option: exit()
    #    workflowdir             = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/workflow/"
    # reading config file
    configurationFile = 'BeamSpotWorkflow.cfg'
    configuration     = ConfigParser.ConfigParser()
    print 'Reading configuration from ', configurationFile
    configuration.read(configurationFile)

    fromDir     = configuration.get('Common','FROM_DIR')
    toDir       = configuration.get('Common','TO_DIR')
    databaseTag = configuration.get('Common','DBTAG')
    dataSet     = configuration.get('Common','DATASET')
    if fromDir[len(fromDir)-1] != '/':
        fromDir = fromDir + '/'
    if toDir[len(toDir)-1] != '/':
        toDir = toDir + '/'
    if not os.path.isdir(toDir):
	os.mkdir(toDir)
    


#    lastUploadedIOV = getLastUploadedIOV(databaseTag) 
    lastUploadedIOV = 133918
    newRunList      = getNewRunList(fromDir,lastUploadedIOV)
    if len(newRunList) == 0:
        exit("There are no new runs after " + lastUploadedIOV)

    listOfFilesToProcess = getListOfFilesToProcess(dataSet,lastUploadedIOV) 

    listOfFilesToCopy = getListOfFilesToCopy(listOfFilesToProcess,newRunList)

    cp(fromDir,toDir,listOfFilesToCopy)    

