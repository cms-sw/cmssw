#!/usr/bin/env python
#G.Benelli Jan 22 2007, J. Nicolson 12 Aug 2008
#A little script to move Simulation Performance Suite
#relevant html and log files into our public area
#/afs/cern.ch/cms/sdt/web/performance/simulation/
#Set here the standard number of events (could become an option... or could be read from the log...)

###############################
#
# Warning!!! We should use copytree() instead of our self defined copytree4
#            However, copytree does not use ignore patterns (for filtering files)
#            before python v2.6, when we upgrade to python 2.6 we should use this
#            functionality.
import tempfile as tmp
import optparse as opt
import cmsPerfRegress as cpr
import re, os, sys, time, glob, socket, fnmatch
from shutil import copy2, copystat
from stat   import *
from cmsPerfCommons import CandFname, Step, ProductionSteps, Candles
import ROOT
#Switching from os.popen4 to subprocess.Popen for Python 2.6 (340_pre5 onwards):
import subprocess

PROG_NAME  = os.path.basename(sys.argv[0])
DEF_RELVAL = "/afs/cern.ch/cms/sdt/web/performance/RelVal"
DEF_SIMUL  = "/afs/cern.ch/cms/sdt/web/performance/simulation"
TMP_DIR    = ""
cpFileFilter = ( "*.root", ) # Unix pattern matching not regex
cpDirFilter  = (           ) # Unix pattern matching not regex

TimeSizeNumOfEvents = -9999
IgProfNumOfEvents   = -9999
CallgrindNumOfEvents = -9999
MemcheckNumOfEvents = -9999

DirName=( #These need to match the candle directory names ending (depending on the type of profiling)
          "TimeSize",
          "IgProf",
          "IgProf_Perf",
          "IgProf_Mem",
          "Callgrind",
          "Memcheck",
          #Adding the extra PU directories:
          "PU_TimeSize",
          "PU_IgProf",
          "PU_IgProf_Perf",
          "PU_IgProf_Mem",
          "PU_Callgrind",
          "PU_Memcheck"
          )
#Defining Steps as a union of Step and ProductionSteps:
Steps=set(Step+ProductionSteps+["GEN,FASTSIM","GEN,FASTSIM_PILEUP"]) #Adding GEN,FASTSIM too by hand.
print Steps

##################
#
# Small functions
#

class ReldirExcept(Exception):
    "Relative directory could not be determined"

def fail(errstr=""):
    print errstr
    delTmpDir()
    sys.exit()

def addtrailingslash(adir):
    trail = re.compile("/$")
    if os.path.isdir(adir) and not trail.search(adir):
        adir = adir + "/" 
    return adir

def getDate():
    return time.ctime()

def getcmdBasic(cmd):
    #Obsolete popen4-> subprocess.Popen
    #return os.popen4(cmd)[1].read().strip()
    return subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read().strip()

def getcmd(command):
    if _debug > 2:
        print command
    #Obsolete popen4-> subprocess.Popen
    #return os.popen4(command)[1].read().strip()
    return subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read().strip()

def prettySize(size):
    nega = size < 0
    if nega:
        size = -size
    suffixes = [("B",2**10), ("k",2**20), ("M",2**30), ("G",2**40), ("T",2**50)]
    for suf, lim in suffixes:
        if size > lim:
            continue
        else:
            if nega:
                return "-" + round(size/float(lim/2**10),2).__str__() + suf
            else:
                return round(size/float(lim/2**10),2).__str__()+suf                

class Row(object):

    def __init__(self,table):
        self.table   = table
        self.coldata = {}
        
    def __str__(self):
        return str(self.coldata)
    
    def addEntry(self,colname,value):
        self.table._newCol(colname)
        self.coldata[colname] = value

    def getRowDict(self):
        return self.coldata

class Table(object):
    
    def __init__(self):
        self.colNames = []
        self.keys     = [None]
        self.rows     = {None: self.colNames}

    def __str__(self):
        out = self.keys
        out += "\n" + self.rows
        return out

    def _newCol(self,name):
        if name in self.colNames:
            pass
        else:
            self.colNames.append(name)

    def getCols(self):
        return self.colNames

    def newRow(self,name):
        if name in self.rows.keys():
            pass
        else:
            self.keys.append(name)
            self.rows[name] = Row(self)
            
        return self.rows[name]

    def getTable(self,mode=0):
        name = "Total"
        
        for key in self.keys:
            if  key == None:
                pass
            else:
                total1 = 0
                total2 = 0                
                rowobj  = self.rows[key]
                rowdict = rowobj.getRowDict()
                for col in self.colNames:
                    if col == None:
                        pass
                    elif rowdict.has_key(col) and not col == name:
                        if mode == 1:
                            total1 += rowdict[col]
                        else:
                            (step_tot1, step_tot2) = rowdict[col]
                            total1 += step_tot1
                            total2 += step_tot2
                if mode == 1:
                    rowobj.addEntry(name,total1)                    
                else:
                    rowobj.addEntry(name,(total1,total2))
                
        return (self.keys, self.rows)

    def addRow(self,row,name):
        if name in self.rows.keys():
            pass
        else:
            self.keys.append(name)
            self.rows[name] = row
            
    def transpose(self):
        transp = Table()
        for col in self.colnames:
            rowobj = transp.newRow(col)
            for key in self.keys:
                if key == None:
                    pass
                else:
                    row_dict = self.rows[key].getRowDict()
                    if row_dict.has_key(key):
                        rowobj.addEntry(key,row_dict[col])
        return transp

######################
#
# Main 
#
def main():
    global TimeSizeNumOfEvents,IgProfNumOfEvents,CallgrindNumOfEvents,MemcheckNumOfEvents
    #Bad design... why is this function defined here?
    #Either no function, or define somewhere else.
    def _copyReportsToStaging(repdir,LogFiles,cmsScimarkDir,stage):
        """Use function syscp to copy LogFiles and cmsScimarkDir over to staging area"""
        if _verbose:
            print "Copying the logfiles to %s/." % stage
            print "Copying the cmsScimark2 results to the %s/." % stage  

        syscp(LogFiles     , stage + "/")
        syscp(cmsScimarkDir, stage + "/")

    #Same comment as above about the opportunity of this function definition here.
    def _createLogFile(LogFile,date,LocalPath,ShowTagsResult):
        """Creating a small logfile with basic publication script running information (never used really by other scripts in the suite)."""
        try:
            LOG = open(LogFile,"w")
            if _verbose:
                print "Writing Production Host, Location, Release and Tags information in %s" % LogFile 
            LOG.write("These performance tests were executed on host %s and published on %s" % (HOST,date))
            LOG.write("They were run in %s" % LocalPath)
            LOG.write("Results of showtags -r in the local release:\n%s" % ShowTagsResult)
            LOG.close()
        except IOError, detail:
            print "WARNING: Can't create log file"            
            print detail

    # Print Program header
    print_header()

    # Get environment variables
    #FIXME: should check into this and make sure the proper variables for the tests being published are read from logfile (case of deprecated releases...)
    print "\n Getting Environment variables..."
    (LocalPath, ShowTagsResult) = get_environ()

    # Parse options
    (options,args) = optionparse()

    # Determine program parameters and input/staging locations
    print "\n Determining locations for input and staging..."
    (drive,path,remote,stage,port,repdir,prevrev,igprof_remotedir) = getStageRepDirs(options,args)

    #Get the number of events for each test from logfile:
    print "\n Getting the number of events for each test..."
    #Let's do a quick implementation of something that looks at the logfile:
    cmsPerfSuiteLogfile="%s/cmsPerfSuite.log"%repdir

    if os.path.exists(cmsPerfSuiteLogfile):
        try:
            (TimeSizeNumOfEvents,IgProfNumOfEvents,CallgrindNumOfEvents,MemcheckNumOfEvents)=getNumOfEventsFromLog(cmsPerfSuiteLogfile)
            #Get the CMSSW version and SCRAM architecture from log (need these for the IgProf new publishing with igprof-navigator)
            (CMSSW_arch,CMSSW_version)=getArchVersionFromLog(cmsPerfSuiteLogfile)
            #For now keep the dangerous default? Better set it to a negative number...
        except:
            print "There was an issue in reading out the number of events for the various tests or the architecture/CMSSW version using the standard logfile %s"%cmsPerfSuiteLogFile
            print "Check that the format was not changed: this scripts relies on the initial perfsuite arguments to be dumped in the logfile one per line!"
            print "For now taking the default values for all tests (0)!"

    print "\n Scan report directory..."
    # Retrieve some directories and information about them
    (ExecutionDate,LogFiles,date,cmsScimarkResults,cmsScimarkDir) = scanReportArea(repdir)
    print "cmsScimarkResults are %s"%cmsScimarkResults
    print "\n Copy report files to staging directory..."
    # Copy reports to staging area 
    _copyReportsToStaging(repdir,LogFiles,cmsScimarkDir,stage)

    print "\n Creating log file..."
    # Produce a small logfile with basic info on the Production area
    _createLogFile("%s/ProductionLog.txt" % stage,date,repdir,ShowTagsResult)

    #Check if there are IgProf tests:
    for dirname in os.listdir(repdir):
        if "IgProf" in dirname:
            print "\n Handling IgProf reports..."
            # add a function to handle the IgProf reports
            stageIgProfReports(igprof_remotedir,CMSSW_arch,CMSSW_version)
    
    print "\n Creating HTML files..."
    # create HTML files
    createWebReports(stage,repdir,ExecutionDate,LogFiles,cmsScimarkResults,date,prevrev)

    print "\n Copy profiling logs to staging directory..."
    # Copy over profiling logs...
    getDirnameDirs(repdir,stage)

    # Send files to remote location
    if remote:
        print "\n Uploading web report to remote location..."
        syncToRemoteLoc(stage,drive,path,port)
        print "\n Finished uploading! Now removing staging directory..."
        delTmpDir()

    print "\n Finished!!!"

##########################
#
# Get require environment variables
#
def get_environ():
    global CMSSW_VERSION, CMSSW_RELEASE_BASE, CMSSW_BASE, HOST, USER, BASE_PERFORMANCE, CMSSW_WORK
    global DEF_RELVAL, DEF_SIMUL
    
    try:
        CMSSW_VERSION=os.environ['CMSSW_VERSION']
        CMSSW_RELEASE_BASE=os.environ['CMSSW_RELEASE_BASE']
        CMSSW_BASE=os.environ['CMSSW_BASE']
        HOST=os.environ['HOST']
        USER=os.environ['USER']
        CMSSW_WORK = os.path.join(CMSSW_BASE,"work/Results")
    except KeyError, detail:
        fail("ERROR: Could not retrieve some necessary environment variables. Have you ran scramv1 runtime -csh yet?. Details: %s" % detail)

    LocalPath=getcmdBasic("pwd")
    ShowTagsResult=getcmdBasic("showtags -r")

    #Adding a check for a local version of the packages
    PerformancePkg="%s/src/Validation/Performance"        % CMSSW_BASE
    if (os.path.exists(PerformancePkg)):
        BASE_PERFORMANCE=PerformancePkg
        print "**Using LOCAL version of Validation/Performance instead of the RELEASE version**"
    else:
        BASE_PERFORMANCE="%s/src/Validation/Performance"  % CMSSW_RELEASE_BASE

    return (LocalPath,ShowTagsResult)

###############
#Option parser
#
def optionparse():
    global PROG_NAME, _debug, _dryrun, _verbose

    parser = opt.OptionParser(usage=("""%s [HOST:]DEST_PATH [Options]

    Arguments:
        [HOST:]DEST_PATH - This is where the report should be published, you can specify a local path or a directory on a remote machine (HOST is optional)

    Examples:  
       Publish report to default local directory
        ./%s 
       Publish report to \"/some/local/dir\"
        ./%s /some/local/dir
       Publish report to \"/some/other/dir\" remote host \"hal.cern.ch\" 
        ./%s hal.cern.ch:/some/other/dir
       Publish report to default relval location (this could be remote or local depending on the hardcoded default)
        ./%s --relval"""
      % ( PROG_NAME, PROG_NAME, PROG_NAME, PROG_NAME, PROG_NAME)))
    
    devel  = opt.OptionGroup(parser, "Developer Options",
                                     "Caution: use these options at your own risk."
                                     "It is believed that some of them bite.\n")
    parser.add_option(
        '--relval',
        action="store_true",
        dest='relval',
        help='Use the default RelVal location',
        #metavar='<STEPS>',
        )

    parser.add_option(
        '-v',
        '--verbose',
        action="store_true",
        dest='verbose',
        help='output more information',
        #metavar='<STEPS>',
        )    

    parser.add_option(
        '--simul',
        action="store_true",
        dest='simulation',
        help='Use the default simulation location',
        #metavar='<STEPS>',
        )

    parser.add_option(
        '--prev',
        type="string",
        dest='previousrev',
        help='The override the name of the previous release. Default is the string obtain from the identification file REGRESSION.<prevrel>.vs.<newrel>',
        metavar='<NAME>',
        default="",
        )    

    parser.add_option(
        '--input',
        type="string",
        dest='repdir',
        help='The location of the report files to be published',
        metavar='<DIR>'
        )

    parser.add_option(
        '-p',
        '--port',
        type='int',
        dest='port',
        help='Use a particular port number to rsync material to a remote server',
        metavar='<PORT>'
        )
    parser.add_option(
        '--igprof',
        type='string',
        dest='ig_remotedir',
        default='IgProfData', #Reverting to local default publication for now#'/afs/cern.ch/cms/sdt/web/qa/igprof-testbed/data', #For now going straight into AFS... later implement security via local disk on cmsperfvm and cron job there:
        #default='cmsperfvm:/data/projects/conf/PerfSuiteDB/IgProfData', #This should not change often! In this virtual machine a cron job will run to move stuff to AFS.
        help='Specify an AFS or host:mydir remote directory instead of default one',
        metavar='<IGPROF REMOTE DIRECTORY>'
        )
    
    devel.add_option(
        '-d',
        '--debug',
        type='int',
        dest='debug',
        help='Show debug output',
        #metavar='DEBUG',
        )

    devel.add_option(
        '--dry-run',
        action="store_true",
        dest='dryrun',
        help='Do not send files to remote server, but run everything else',
        #metavar='DEBUG',
        )

    repdirdef = os.getcwd()
    parser.set_defaults(debug=0,simulation=False,relval=False,port=873,pretend=False,repdir=repdirdef,verbose=False)
    parser.add_option_group(devel)

    (options, args) = parser.parse_args()

    _debug   = options.debug
    _dryrun  = options.dryrun
    _verbose = options.verbose

    numofargs = len(args) 

    if (options.simulation and options.relval) or ((options.simulation or options.relval) and numofargs >= 1):
        parser.error("You can not specify simulation and relval together. Neither can you specify simulation or relval AND a path")
        sys.exit()

    return (options, args)


#A function to get the number of events for each test from the logfile:
def getNumOfEventsFromLog(logfile):
    '''A very fragile function to get the Number of events for each test by parsing the logfile of the Suite. This relies on the fact that nobody will turn off the print out of the options in the cmsPerfSuite.py output... ARGH!'''
    log=open(logfile,"r")
    TimeSizeEvents=0
    IgProfEvents=0
    CallgrindEvents=0
    MemcheckEvents=0
    for line in log:
        #FIXME:
        #For robustness could read this from the Launching the X tests (.....) with N events each and keep that format decent for parsing.
        #One more place where XML would seem a better choice to extract information (Timing of the performance suite in general is also, but publishing and harvesting some other info as well).
        if 'TimeSizeEvents' in line and not TimeSizeEvents:
            lineitems=line.split()
            TimeSizeEvents=lineitems[lineitems.index('TimeSizeEvents')+1]
        if 'IgProfEvents' in line and not IgProfEvents:
            lineitems=line.split()
            IgProfEvents=lineitems[lineitems.index('IgProfEvents')+1]
        if 'CallgrindEvents' in line and not CallgrindEvents:
            lineitems=line.split()
            CallgrindEvents=lineitems[lineitems.index('CallgrindEvents')+1]
        if 'MemcheckEvents' in line and not MemcheckEvents:
            lineitems=line.split()
            MemcheckEvents=lineitems[lineitems.index('MemcheckEvents')+1]
    return (TimeSizeEvents,IgProfEvents,CallgrindEvents,MemcheckEvents)

def getArchVersionFromLog(logfile):
    '''Another very fragile function to get the architecture and the CMSSW version parsing the logfile...'''
    log=open(logfile,"r")
    arch=re.compile("^Current Architecture is")
    version=re.compile("^Current CMSSW version is")
    CMSSW_arch="UNKNOWN_ARCH"
    CMSSW_version="UNKNOWN_VERSION"
    for line in log:
        if arch.search(line):
            CMSSW_arch=line.split()[3]
        if version.search(line):
            CMSSW_version=line.split()[4]
    return(CMSSW_arch,CMSSW_version)


#####################
#
# Determine locations of staging and report dirs 
#
def getStageRepDirs(options,args):
    global TMP_DIR, IS_TMP, DEF_LOCAL, CMSSW_VERSION
    DEF_LOCAL = CMSSW_WORK
    numofargs = len(args)

    repdir = os.path.abspath(options.repdir)
    repdir = addtrailingslash(repdir)

    if not os.path.exists(repdir):
        fail("ERROR: The specified report directory %s to retrieve report information from does not exist, exiting" % repdir)

    previousrev = options.previousrev
    if previousrev == "":
        regressfiles = glob.glob("%s/REGRESSION.*.vs.*" % repdir)
        if not len(regressfiles) == 0:
            regressID   = regressfiles[0]
            base        = os.path.basename(regressID)
            split       = base.split(".")
            previousrev = split[1]
            currentrel  = split[3]
            print "Regression Identification file exists, renaming report title for regression report. Old ver: %s" % previousrev
        else:
            print "No regression ID file exists and previous release name was not specified. Producing normal report."
    else:
        print "Previous release name was specified, renaming report title for regression report. Old ver %s" % previousrev
    
    uri = ""
    defaultlocal = False
    if options.simulation:
        uri = DEF_SIMUL
    elif options.relval:
        uri = DEF_RELVAL
    elif numofargs >= 1:
        uri = args[0] # Add directory CMSSW_VERSION later in temp! Not now, otherwise we get into a mess if this is a remote dir
    else:
        defaultlocal = True
        uri = DEF_LOCAL

    ####
    #
    # Determine if location is remote
    #
    # Try not to re-arrange we don't want to assume that default locations are not remote
    #
    ####

    drive = ""
    path = ""
    if ":" in uri:
        drive, path = uri.split(":",1)
    else:
        path = uri
        
    if drive == "":
        path = os.path.abspath(path)
    remote = not drive == ""

    if remote:
        unResolved = True
        try:
            socket.getaddrinfo(drive,53)
            unResolved = False
        except socket.gaierror:
            unResolved = True

        # try see if it's an ipaddress
        if unResolved:
            try:
                socket.gethostbyaddr(drive)
                unResolved = False
            except socket.gaierror:
                unResolved = True
            if unResolved:
                print "ERROR: Can not determine your hostname or ipv{4,6} address %s" % drive
                if not (_dryrun or _test):
                    fail("exiting...")

    if (not remote) and (not options.port == 873) :
        print "WARNING: Can not use a port if not performing a remote copy, ignoring"
    port = options.port

    ###
    #
    # Determine Staging Area
    #

    StagingArea=""
    localExists = os.path.exists("%s/%s" % (CMSSW_WORK,CMSSW_VERSION))
    
    if remote:
        #Cannot use this since the /tmp is limited to 2GB on lxbuild machines!
        #TMP_DIR=tmp.mkdtemp(prefix="/tmp/%s" % PROG_NAME)
        TMP_DIR=tmp.mkdtemp(prefix="/build/%s" % PROG_NAME)
        StagingArea = TMP_DIR
    #Local but dir already exists
    elif defaultlocal and localExists:
        TMP_DIR=tmp.mkdtemp(prefix="%s/%s" % (CMSSW_WORK,CMSSW_VERSION))
        StagingArea = TMP_DIR
        print "WARNING: %s already exists, creating a temporary staging area %s" % (CMSSW_WORK,TMP_DIR)
    #Local cases
    elif defaultlocal:
        StagingArea = CMSSW_WORK
        try:
            os.mkdir(os.path.join(CMSSW_BASE,"work"))
            os.mkdir(os.path.join(CMSSW_BASE,"work","Results"))
        except OSError:
            pass
        print "**User did not specify location of results, staging in default %s**" % StagingArea 
    else:
        print "**User chose to publish results in a local directory**" 
        StagingArea = path
        if not os.path.exists(path):
            try:
                os.mkdir("%s" % path)
            except OSError, detail:
                if   detail.errno == 13:
                    fail("ERROR: Failed to create staging area %s because permission was denied " % StagingArea)
                elif detail.errno == 17:
                    #If the directory already exists just carry on
                    pass
                else:
                    fail("ERROR: There was some problem (%s) when creating the staging directory" % detail)            

    IS_TMP = not TMP_DIR == ""
    ######
    #
    # create Version dir
    # This is why when we create a Tmp dir we get the version(tmpname)/version
    # structure. We should remove this if we don't want it but i dont think it matters
    StagingArea="%s/%s" % (StagingArea,CMSSW_VERSION)
    try:
        os.mkdir("%s" % StagingArea)
    except OSError, detail:
        if   detail.errno == 13:
            fail("ERROR: Failed to create staging area %s because permission was denied " % StagingArea)
        elif detail.errno == 17:
            #If the directory already exists just carry on
            pass
        else:
            fail("ERROR: There was some problem (%s) when creating the staging directory" % detail)
    
    return (drive,path,remote,StagingArea,port,repdir,previousrev,options.ig_remotedir)

####################
#
# Scan report area for required things
#
def scanReportArea(repdir):
    """Scans the working directory for cms*.logs (cmsPerfSuite.log and cmsScimark*.log, and cmsScimark results.
    It returns Execution date (completion), current date, list of logfiles and cmsScimark results"""
    date=getDate()
    LogFiles  = glob.glob(repdir + "cms*.log")
    if _verbose:
        print "Found the following log files:"
        print LogFiles

    cmsScimarkDir = glob.glob(repdir + "cmsScimarkResults_*")
    if _verbose:
        print "Found the following cmsScimark2 results directories:"
        print cmsScimarkDir

    cmsScimarkResults = []
    for adir in cmsScimarkDir:
        htmlfiles = glob.glob(adir + "/*.html")
        #FIXME:
        #Really unnecessary use of map in my opinion
        #Could do with cmsScimarkResults.extend(htmlfiles)
        map(cmsScimarkResults.append,htmlfiles)

    ExecutionDateLast = ""
    ExecutionDate = ""
    ExecutionDateSec=0
    cmsreg = re.compile("^cmsPerfSuite")
    for logf in LogFiles:
        if cmsreg.search(logf):
            ExecutionDateLastSec = os.stat(logf)[ST_CTIME]
            ExecutionDateLast    = os.stat(logf)[ST_MTIME]
            if _verbose:
                print "Execution (completion) date for %s was: %s" % (logf,ExecutionDateLast)
            if (ExecutionDateLastSec > ExecutionDateSec):
                ExecutionDateSec = ExecutionDateLastSec
                ExecutionDate    = ExecutionDateLast
                
    if ExecutionDate == "":
        ExecutionDate = ExecutionDateLast

    return (ExecutionDate,LogFiles,date,cmsScimarkResults,cmsScimarkDir)

def createRegressHTML(reghtml,repdir,outd,CurrentCandle,htmNames):
    RegressTmplHTML="%s/doc/regress.html" % (BASE_PERFORMANCE)
    candnreg  = re.compile("CandleName")
    candhreg  = re.compile("CandlesHere")
    try:
        REGR = open(reghtml,"w")
        for line in open(RegressTmplHTML):
            if candhreg.search(line):
                html = "<table>"
                
                for x in htmNames:
                    abspath = os.path.join(repdir,outd)
                    if os.path.exists(abspath):
                        html += "<tr><td><a href=\"./%s/%s\"><img src=\"./%s/%s\" /></a></td></tr>\n" % (outd,x,outd,x)
                    else:
                        html += "<tr><td> %s does not exist probably because the log file for the previous release was missing</td></tr>" % (abspath)
                html += "</table>"
                REGR.write(html)
            elif candnreg.search(line):
                REGR.write(CurrentCandle)
            else:
                REGR.write(line)
    except IOError, detail:
        print "ERROR: Could not write regression html %s because %s" % (os.path.basename(reghtml),detail)                

def getOutputNames(base,reportName):
    logreg   = re.compile("(.*)\.log$")
    matches  = logreg.search(reportName)
    logdir   = logreg.sub(matches.groups()[0],reportName)
    outd     = os.path.join(base,logdir)
    nologext = matches.groups()[0]
    return (nologext,outd)

def rootfile_cmp(x,y):
    (fname,x) = x
    (fname,y) = y
    x = os.path.basename(x)
    y = os.path.basename(y)
    stepreg = re.compile("%s_(..*)\.root" % fname)
    if stepreg.search(x):
        x = stepreg.search(x).groups()[0]
    if stepreg.search(y):
        y = stepreg.search(y).groups()[0]
    return step_cmp(x,y)

def dirname_cmp(x,y):
    (candle,prof,x) = x
    (candle,prof,y) = y
    x = os.path.basename(x)
    y = os.path.basename(y)
    stepreg = re.compile("%s_(..*)_%s" % (candle,prof))
    if stepreg.search(x):
        x = stepreg.search(x).groups()[0]
    if stepreg.search(y):
        y = stepreg.search(y).groups()[0]
    return step_cmp(x,y)

def reg_dirname_cmp(x,y):
    (candle,prof,x) = x
    (candle,prof,y) = y
    x = os.path.basename(x)
    y = os.path.basename(y)
    stepreg = re.compile("%s_(..*)_%s_regression" % (candle,prof))
    if stepreg.search(x):
        x = stepreg.search(x).groups()[0]
    if stepreg.search(y):
        y = stepreg.search(y).groups()[0]
    return step_cmp(x,y)

def logrep_cmp(x,y):
    (fname,x) = x
    (fname,y) = y    
    x = os.path.basename(x)
    y = os.path.basename(y)
    stepreg = re.compile("%s_(..*)_TimingReport.log" % fname)
    if stepreg.search(x):
        x = stepreg.search(x).groups()[0]
    if stepreg.search(y):
        y = stepreg.search(y).groups()[0]
    return step_cmp(x,y)

def timerep_cmp(x,y):
    (fname,x) = x
    (fname,y) = y        
    x = os.path.basename(x)
    y = os.path.basename(y)
    stepreg = re.compile("%s_(..*)_TimingReport" % fname)
    if stepreg.search(x):
        x = stepreg.search(x).groups()[0]
    if stepreg.search(y):
        y = stepreg.search(y).groups()[0]
    return step_cmp(x,y)

def step_cmp(x,y):
    xstr = x
    ystr = y
    x_idx = -1
    y_idx = -1
    bestx_idx = -1
    besty_idx = -1
    sndbst_x = -1
    sndbst_y = -1
    last_x = -1
    last_y = -1        
    for i in range(len(Step)):
        stepreg = re.compile("^%s.*" % Step[i])
        # fallback
        if Step[i] in xstr and sndbst_x == -1:         
            last_x = i
        if Step[i] in ystr and sndbst_y == -1:                     
            last_y = i        
        # second best
        if xstr in Step[i] and bestx_idx == -1:         
            sndbst_x = i
        if ystr in Step[i] and besty_idx == -1:                     
            sndbst_y = i
        # next best
        if stepreg.search(xstr) and x_idx == -1: 
            bestx_idx = i # If an exact match has not been found but something similar has, set x best index
        if stepreg.search(ystr) and y_idx == -1:
            besty_idx = i # If an exact match has not been found but something similar has, set y best index
        # actual
        if Step[i] == xstr and x_idx == -1:
            x_idx = i     # if an exact match has been found then set x index
        if Step[i] == ystr and y_idx == -1:
            y_idx = i     # if an exact match has been found then set y index
        if not ( x_idx == -1 or y_idx == -1):
            break         # if an exact match has been found for both, stop

    # use best match if we still can't find indices
    if x_idx == -1:
        x_idx = bestx_idx
    if y_idx == -1:
        y_idx = besty_idx

    # use second best if we still can't find indices
    if x_idx == -1:
        x_idx = sndbst_x
    if y_idx == -1:
        y_idx = sndbst_y

    # use fallback if we still can't find indices
    if x_idx == -1:
        x_idx = last_x
    if y_idx == -1:
        y_idx = last_y

    if x_idx == -1 or y_idx == -1:
        print "WARNING: No valid step names could be found in the logfiles or root filenames being sorted: x: %s y: %s." % (xstr,ystr)
        print "x", x_idx, "y", y_idx

    if x_idx < y_idx:
        return -1
    elif x_idx == y_idx:
        return 0
    elif y_idx < x_idx:
        return 1
        

######################
#
# Create HTML pages for candles

def createCandlHTML(tmplfile,candlHTML,CurrentCandle,WebArea,repdir,ExecutionDate,LogFiles,cmsScimarkResults,date,prevrev):
    global TimeSizeNumOfEvents,IgProfNumOfEvents,CallgrindNumOfEvents,MemcheckNumOfEvents
    def _stripbase(base, astr):
        basereg = re.compile("^%s/?(.*)" % base)
        out = astr
        found = basereg.search(astr)
        if found:
            out = found.groups()[0]
        return out
    
    def _getProfileReportLink(repdir,CurrentCandle,CurDir,step,CurrentProfile,Profiler):
        #FIXME:
        #Pileup now has it own directory... should add it in the DirName dictionary at the beginning?
        ProfileTemplate=os.path.join(repdir, "%s_%s" % (CurrentCandle,CurDir), "*_%s_%s*" % (step,CurrentProfile),Profiler)
        #print ProfileTemplate
        #There was the issue of SIM vs sim (same for DIGI) between the previous RelVal based performance suite and the current.
        ProfileTemplateLowCaps=os.path.join(repdir, "%s_%s" % (CurrentCandle,CurDir), "*_%s_%s*" % (step.lower(),CurrentProfile),Profiler)        
        ProfileReportLink = glob.glob(ProfileTemplate)
        #Filter out the Pile up directories when step does not contain Pile up
        #if not ('PILEUP' in step):
        #    print "NPNPNPNP BEFORE %s"%ProfileReportLink
        #    ProfileReportLink=filter(lambda x: "PU" not in x,ProfileReportLink)
        #    print "NPNPNPNP %s"%ProfileReportLink
        #print ProfileReportLink
        if len(ProfileReportLink) > 0:
            #print ProfileReportLink
            if not reduce(lambda x,y: x or y,map(lambda x: CurrentCandle in x,ProfileReportLink)):# match with caps try low caps
                ProfileReportLink = glob.glob(ProfileTemplateLowCaps)
        else:
            ProfileReportLink = glob.glob(ProfileTemplateLowCaps)
        ProfileReportLink = map(lambda x: _stripbase(repdir,x),ProfileReportLink)
        #print ProfileReportLink
        return ProfileReportLink

    def _writeReportLink(INDEX,ProfileReportLink,CurrentProfile,step,NumOfEvents,Profiler=""):
        if Profiler == "":
            INDEX.write("<li><a href=\"%s\">%s %s (%s events)</a></li>\n" % (ProfileReportLink,CurrentProfile,step,NumOfEvents))
        else:
            #FIXME: need to fix this: why do we have the number of Memcheck events hardcoded? 
            if CurrentProfile == "memcheck_valgrind":#FIXME:quick and dirty hack to report we have changed memcheck to 5 events now...
                INDEX.write("<li><a href=\"%s\">%s %s %s (%s events)</a></li>\n" % (ProfileReportLink,CurrentProfile,Profiler,step,"5"))
            else:
                INDEX.write("<li><a href=\"%s\">%s %s %s (%s events)</a></li>\n" % (ProfileReportLink,CurrentProfile,Profiler,step,NumOfEvents))
    def IgProfDumpsTable(INDEX,ProfileLinks,step):
        #See if the end of job profiles IgProfMemTotal.res or IgProfMemLive.res are in the list as they should:
        EndOfJobProfileLink=filter(lambda x: "IgProfMemTotal.res" in x or "IgProfMemLive.res" in x, ProfileLinks)[0] #Assume only one of the two is there, as it should.
        #Remove it from the list so we can order it:
        ProfileLinks.remove(EndOfJobProfileLink)
        #Sort the list in numerical order:
        ProfileLinks.sort(key=lambda x: int(x.split(".")[-2]))
        #Prepare regexp to find and replace MemTotal with MemLive for .gz links
        IgProfMemLive_regexp=re.compile("IgProfMemLive")
        if IgProfMemLive_regexp.search(EndOfJobProfileLink):
            MemProfile="IgProf MEM LIVE"
        else:
            MemProfile="IgProf MEM TOTAL"
        #Prepare the table header:
        INDEX.write("<li>%s"%MemProfile)
        INDEX.write("<table><tr><td>Profile after event</td><td>Total Memory Size (bytes)</td><td>Total Calls (number)</td><td>Link to gzipped IgProf profile</td></tr>")
        for link in ProfileLinks:
            #Build and check the link to the .gz profile that is always called IgProfMemTotal also for MemLive:
            gzProfile=os.path.join(link.split("/")[-3],link.split("/")[-1])[:-3]+"gz"
            if IgProfMemLive_regexp.search(gzProfile):
                gzProfile=IgProfMemLive_regexp.sub(r"IgProfMemTotal",gzProfile)
            INDEX.write("<tr><td>%s</td><td>%s</td><td>%s</td><td><a href=%s>%s</a></td></tr>"%(link.split(".")[-2],open(link,"r").readlines()[6].split()[1],open(link,"r").readlines()[6].split()[2],gzProfile,os.path.basename(gzProfile)))
        #Put in the end of job one by hand:
        gzEndOfJobProfileLink=os.path.join(EndOfJobProfileLink.split("/")[-3],EndOfJobProfileLink.split("/")[-1])[:-3]+"gz"
        if IgProfMemLive_regexp.search(gzEndOfJobProfileLink):
            gzEndOfJobProfileLink=IgProfMemLive_regexp.sub(r"IgProfMemTotal",gzEndOfJobProfileLink)
        INDEX.write("<tr><td>%s</td><td>%s</td><td>%s</td><td><a href=%s>%s</a></td></tr>"%("End of job",open(EndOfJobProfileLink,"r").readlines()[6].split()[1],open(EndOfJobProfileLink,"r").readlines()[6].split()[2],gzEndOfJobProfileLink,os.path.basename(gzEndOfJobProfileLink)))
        #Closing the table and the list item tags
        INDEX.write("</table>")
        INDEX.write("</li>")
    #FIXME:
    #These numbers are used in the index.html they are not automatically matched to the actual
    #ones (one should automate this, by looking into the cmsCreateSimPerfTestPyRelVal.log logfile)    
                            
    NumOfEvents={
                DirName[0] : TimeSizeNumOfEvents,
                DirName[1] : IgProfNumOfEvents,
                DirName[2] : IgProfNumOfEvents,
                DirName[3] : IgProfNumOfEvents,
                DirName[4] : CallgrindNumOfEvents,
                DirName[5] : MemcheckNumOfEvents,
                DirName[6] : TimeSizeNumOfEvents,
                DirName[7] : IgProfNumOfEvents,
                DirName[8] : IgProfNumOfEvents,
                DirName[9] : IgProfNumOfEvents,
                DirName[10] : CallgrindNumOfEvents,
                DirName[11] : MemcheckNumOfEvents
                }

    Profile=(
        #These need to match the profile directory names ending within the candle directories
              "TimingReport",
              "TimeReport",
              "SimpleMemReport",
              "EdmSize",
              "IgProfperf",
              "IgProfMemTotal",
              "IgProfMemLive",
              "IgProfMemAnalyse",
              "valgrind",
              "memcheck_valgrind"
               )
    IgProfMemAnalyseOut=( #This is the special case of IgProfMemAnalyse
                          "doBeginJob_output.html",
                          "doEvent_output.html",
                          "mem_live.html",
                          "mem_total.html" 
                          )
    memcheck_valgrindOut=( #This is the special case of Valgrind MemCheck (published via Giovanni's script)
                           "beginjob.html",
                           "edproduce.html",
                           "esproduce.html"
                           )
    OutputHtml={ #These are the filenames to be linked in the candle html page for each profile
                 Profile[0] : "*TimingReport.html", #The wildcard spares the need to know the candle name
                 Profile[1] : "TimeReport.html", #This is always the same (for all candles)
                 Profile[2] : "*.html", #This is supposed to be *SimpleMemoryCheck.html, but there is a bug in cmsRelvalreport.py and it is called TimingReport.html!
                 Profile[3] : "objects_pp.html", #This is one of 4 objects_*.html files, it's indifferent which one to pick, just need consistency
                 Profile[4] : "overall.html", #This is always the same (for all candles)
                 Profile[5] : "overall.html", #This is always the same (for all candles)
                 Profile[6] : "overall.html", #This is always the same (for all candles)
                 Profile[7] : "doBeginJob_output.html", #This is complicated... there are 3 html to link... (see IgProf MemAnalyse below)
                 Profile[8] : "overall.html", #This is always the same (for all candles)
                 Profile[9] : "beginjob.html" #This is complicated there are 3 html to link here too... (see Valgrind MemCheck below)
                 }


    candnreg  = re.compile("CandleName")
    candhreg  = re.compile("CandlesHere")
    try:
        CAND = open(candlHTML,"w")
        for line in open(tmplfile):
            if candhreg.search(line):
                CAND.write('<div id="content">')                
                CAND.write("<h2>")
                CAND.write(CurrentCandle)
                CAND.write("</h2>\n")

                if _verbose:
                    print "Producing candles html: ", CurrentCandle
                
                for CurDir in DirName:

                    LocalPath = os.path.join(repdir,"%s_%s" % (CurrentCandle,CurDir))
                    LocalDirname = os.path.basename(LocalPath)

                    if not prevrev == "":

                        profs = []
                        #FIXME!
                        #Check what this was used for.. it now has different DirName list since we added IgProf_Perf and IgProf_Mem on top of IgProf alone (still there)
                        if   CurDir == DirName[0] or CurDir == DirName[6]: #TimeSize tests (with and without PU)
                            profs = Profile[0:4]
                        elif CurDir == DirName[1] or CurDir == DirName[7]: #IgProf tests (with and without PU)
                            profs = Profile[4:8]
                        elif CurDir == DirName[2] or CurDir == DirName[8]: #IgProf Perf tests (with and without PU)
                            profs =[Profile[4]] #Keeping format to a list...
                        elif CurDir == DirName[3] or CurDir == DirName[9]: #IgProf Mem tests (with and without PU)
                            profs =Profile[5:8] #Keeping format to a list...  
                        elif CurDir == DirName[4] or CurDir == DirName[10]: #Callgrind tests (with and without PU)
                            profs = [Profile[8]] #Keeping format to a list...
                        elif CurDir == DirName[5] or CurDir == DirName[11]: #Memcheck tests (with and without PU)
                            profs = [Profile[9]] #Keeping format to a list...
                        #This could be optimized, but for now just comment the code:
                        #This for cycle takes care of the case in which there are regression reports to link to the html:
                        for prof in profs:
                            if _verbose:
                                print "Scanning for profile information for: ", prof
                                
                            printed = False
                            fullprof = (CurrentCandle,prof)
                            outd     = ""
                            nologext = ""                            
                            if prof == "TimingReport":
                                timeReports = glob.glob(os.path.join(LocalPath,"%s_*_%s.log" % (CandFname[CurrentCandle],prof)))
                                if len(timeReports) > 0:
                                    if not printed:
                                        CAND.write("<p><strong>%s %s</strong></p>\n" % (prof,"Regression Analysis"))                                        
                                        printed = True
                                    for log in timeReports:
                                        reportName = os.path.basename(log)
                                        (nologext, outd) = getOutputNames(LocalDirname,reportName) 
                                        CAND.write("<h4>%s</h4>\n" % reportName)
                                        htmNames   = ["changes.png"]
                                        otherNames = ["graphs.png" , "histos.png"] 
                                        regressHTML= "%s-regress.html" % nologext
                                        pathsExist = reduce (lambda x,y: x or y,map(os.path.exists,map(lambda x: os.path.join(repdir,outd,x),otherNames)))
                                        html = ""
                                        if pathsExist:
                                            html = "<p>Performance graphs and histograms superimposed for %s are <a href=\"%s\">here</a></p>\n" % (reportName,regressHTML)
                                        else:
                                            html = "<p>No change in performance graph available</p>\n"
                                        regressHTML="%s/%s" % (WebArea,regressHTML)

                                        for x in htmNames:
                                            abspath = os.path.join(repdir,outd,x)
                                            if os.path.exists(abspath):
                                                html += "<p><a href=\"./%s/%s\"><img src=\"./%s/%s\" /></a></p>\n" % (outd,x,outd,x)
                                            else:
                                                html += "<p>%s does not exist probably because the log file for the previous release was missing</p>" % (abspath)

                                        createRegressHTML(regressHTML,repdir,outd,CurrentCandle,otherNames)
                                        CAND.write(html)
                                        CAND.write("\n</tr></table>")                                


                            elif prof == "SimpleMemReport":
                                simMemReports = glob.glob(os.path.join(LocalPath,"%s_*_%s" % (CandFname[CurrentCandle],prof)))
                                simMemReports = map(lambda x: (CandFname[CurrentCandle],prof,x), simMemReports )
                                simMemReports.sort(cmp=dirname_cmp)
                                simMemReports = map(lambda x: x[2], simMemReports )
                                if len(simMemReports) > 0:
                                    if not printed:
                                        CAND.write("<p><strong>%s %s</strong></p>\n" % (prof,"Regression Analysis"))
                                        printed = True                                    
                                    for adir in simMemReports:
                                        reportName = os.path.basename(adir)
                                        CAND.write("<h4>%s</h4>\n" % reportName)
                                        htmNames   = ["vsize_change.png", "rss_change.png"]
                                        otherNames = ["vsize_graphs.png","rss_graphs.png"]
                                        nologext = reportName
                                        outd     = reportName
                                        regressHTML= "%s-regress.html" % nologext
                                        pathsExist = reduce (lambda x,y: x or y,map(os.path.exists,map(lambda x: os.path.join(repdir,LocalDirname,outd,x),otherNames)))
                                        html = ""
                                        if pathsExist:
                                            html = "<p>Superimposed memory performance graphs for %s are <a href=\"%s\">here</a></p>\n" % (reportName,regressHTML)
                                        else:
                                            html = "<p>No change in performance graph available</p>\n"
                                        regressHTML="%s/%s" % (WebArea,regressHTML)

                                        for x in htmNames:
                                            abspath = os.path.join(repdir,LocalDirname,outd,x)
                                            if os.path.exists(abspath):
                                                html += "<p><a href=\"./%s/%s/%s\"><img src=\"./%s/%s/%s\" /></a></p>\n" % (LocalDirname,outd,x,LocalDirname,outd,x)
                                            else:
                                                html += "<p>%s does not exist probably because the log file for the previous release was missing</p>" % (abspath)

                                        createRegressHTML(regressHTML,repdir,"%s/%s" % (LocalDirname,outd),CurrentCandle,otherNames)
                                        CAND.write(html)
                                        CAND.write("\n</tr></table>")   

                            elif prof == "EdmSize" or prof == "IgProfMemTotal" or prof == "IgProfMemLive" or prof== "IgProfperf" or prof == "Callgrind":
                                regresPath = os.path.join(LocalPath,"%s_*_%s_regression" % (CandFname[CurrentCandle],prof))
                                regresses  = glob.glob(regresPath)
                                stepreg = re.compile("%s_([^_]*(_PILEUP)?)_%s_regression" % (CandFname[CurrentCandle],prof))
                                if len(regresses) > 0:
                                    if not printed:
                                        CAND.write("<p><strong>%s %s</strong></p>\n" % (prof,"Regression Analysis"))                                        
                                        printed = True
                                    regresses = map(lambda x: (CandFname[CurrentCandle],prof,x),regresses)                                        
                                    regresses.sort(cmp=reg_dirname_cmp)                                    
                                    regresses = map(lambda x: x[2],regresses)
                                    for rep in regresses:
                                        base  = os.path.basename(rep)
                                        found = stepreg.search(base)
                                        step = "Unknown-step"
                                        if found:
                                            step = found.groups()[0]
                                        htmlpage = ""
                                        if prof == "IgProfMemLive" or prof == "IgProfMemTotal" or prof== "IgProfperf" or prof == "Callgrind":
                                            htmlpage = "overall.html"
                                        else:
                                            htmlpage = "objects_pp.html"
                                        CAND.write("<a href=\"%s/%s/%s\">%s %s regression report</a><br/>\n" % (LocalDirname,base,htmlpage,prof,step))
                    #Here we go back to the "non-regression" reports listing
                    CandleLogFiles = []
                    if os.path.exists(LocalPath):
                        thedir = os.listdir(LocalPath)
                        CandleLogFiles = filter(lambda x: (x.endswith(".log") or x.endswith("EdmSize")) and not os.path.isdir(x) and os.path.exists(x), map(lambda x: os.path.abspath(os.path.join(LocalPath,x)),thedir))                    

                    if (len(CandleLogFiles)>0):
                        
                        syscp(CandleLogFiles,WebArea + "/")
                        base = os.path.basename(LocalPath)
                        lfileshtml = ""
                     
                        for cand in CandleLogFiles:
                            cand = os.path.basename(cand)
                            if _verbose:
                                print "Found %s in %s\n" % (cand,LocalPath)
                                
                            if not "EdmSize" in cand:
                                lfileshtml += "<a href=\"./%s/%s\">%s </a><br/>" % (base,cand,cand)
                                    
                        CAND.write("<p><strong>Logfiles for %s</strong></p>\n" % CurDir)    
                        CAND.write(lfileshtml)

                    PrintedOnce = False
                    for CurrentProfile in Profile:
                        #print Steps
                        for step in Steps: #Using Steps here that is Step+ProductionSteps!
                            #print step
                            #print "DEBUG:%s,%s,%s,%s,%s,%s"%(repdir,CurrentCandle,CurDir,step,CurrentProfile,OutputHtml[CurrentProfile])
                            ProfileReportLink = _getProfileReportLink(repdir,CurrentCandle,
                                                                     CurDir,
                                                                     step,
                                                                     CurrentProfile,
                                                                     OutputHtml[CurrentProfile])
                            #if ProfileReportLink:
                            #    print ProfileReportLink
                            isProfinLink = False
                            if len (ProfileReportLink) > 0:
                                isProfinLink = reduce(lambda x,y: x or y,map(lambda x: CurrentProfile in x,ProfileReportLink))

                            if isProfinLink:
                                #It could also not be there

                                if (PrintedOnce==False): 
                                    #Making sure it's printed only once per directory (TimeSize, IgProf, Valgrind) each can have multiple profiles

                                    #This is the "title" of a series of profiles, (TimeSize, IgProf, Valgrind)
                                    CAND.write("<p><strong>%s</strong></p>\n" % CurDir)
                                    CAND.write("<ul>\n")
                                    PrintedOnce=True
                                
                                #Special cases first (IgProf MemAnalyse and Valgrind MemCheck)
                                #Add among the special cases any IgProfMem (5,6,7) since now we added the dumping:
                                if (CurrentProfile == Profile[5] or CurrentProfile == Profile[6]):
                                    for prolink in ProfileReportLink:
                                        _writeReportLink(CAND,prolink,CurrentProfile,step,NumOfEvents[CurDir])
                                    for igprofmem in ["*IgProfMemTotal*.res","*IgProfMemLive*.res"]:
                                        ProfileReportLink = _getProfileReportLink(repdir,CurrentCandle,
                                                                                 CurDir,
                                                                                 step,
                                                                                 CurrentProfile,
                                                                                 igprofmem)
                                        isProfinLink = False
                                        if len (ProfileReportLink) > 0:
                                            isProfinLink = reduce(lambda x,y: x or y,map(lambda x: CurrentProfile in x,ProfileReportLink))                                        
                                        if isProfinLink :#It could also not be there
                                            #for prolink in ProfileReportLink:
                                            IgProfDumpsTable(CAND,ProfileReportLink,step)
                                            #    _writeReportLink(CAND,prolink,CurrentProfile,step,NumOfEvents[CurDir],Profiler=os.path.basename(prolink))
                                    
                                elif (CurrentProfile == Profile[7]):
                                    for igprof in IgProfMemAnalyseOut:
                                        ProfileReportLink = _getProfileReportLink(repdir,CurrentCandle,
                                                                                 CurDir,
                                                                                 step,
                                                                                 CurrentProfile,
                                                                                 igprof)
                                        isProfinLink = False
                                        if len (ProfileReportLink) > 0:
                                            isProfinLink = reduce(lambda x,y: x or y,map(lambda x: CurrentProfile in x,ProfileReportLink))                                        
                                        if isProfinLink :#It could also not be there
                                            for prolink in ProfileReportLink:
                                                _writeReportLink(CAND,prolink,CurrentProfile,step,NumOfEvents[CurDir],Profiler=igprof)


                                elif (CurrentProfile == Profile[9]):

                                    for memprof in memcheck_valgrindOut:
                                        #print memprof
                                        ProfileReportLink = _getProfileReportLink(repdir,CurrentCandle,
                                                                                  CurDir,
                                                                                  step,
                                                                                  CurrentProfile,
                                                                                  memprof
                                                                                  )
                                        isProfinLink = False
                                        if len (ProfileReportLink) > 0:
                                            isProfinLink = reduce(lambda x,y: x or y,map(lambda x: CurrentProfile in x,ProfileReportLink))                                        
                                        if isProfinLink :#It could also not be there                                        
                                            for prolink in ProfileReportLink:
                                                _writeReportLink(CAND,prolink,CurrentProfile,step,NumOfEvents[CurDir],Profiler=memprof)

                                else:
                                    for prolink in ProfileReportLink:
                                        if "regression" not in prolink: #To avoid duplication of links to regression reports!
                                            _writeReportLink(CAND,prolink,CurrentProfile,step,NumOfEvents[CurDir])
                                            #print "Step is %s, CurrentProfile is %s and ProfileReportLink is %s and prolink is %s"%(step,CurrentProfile,ProfileReportLink,prolink)


                    if PrintedOnce:
                        CAND.write("</ul>\n")
                    PrintedOnce=False

                CAND.write("<hr />")
                CAND.write("<br />\n")
                CAND.write("</div>\n")
            elif candnreg.search(line):
                CAND.write(CurrentCandle)
            else:
                CAND.write(line)

        CAND.close()
    except IOError, detail:
        print "ERROR: Could not write candle html %s because %s" % (os.path.basename(candlHTML),detail)


def populateFromTupleRoot(tupname,repdir,rootfile,pureg):
    table = Table()
    for cand in Candles:
        fname = CandFname[cand]
        globpath = os.path.join(repdir,"%s_TimeSize" % cand,"%s_*_TimingReport" % fname)
        stepDirs = glob.glob(globpath)
        stepDirs = map(lambda x: (fname,x), stepDirs)
        stepDirs.sort(cmp=timerep_cmp)
        stepDirs = map(lambda x: x[1], stepDirs)
        stepreg = re.compile("%s_(.*)_TimingReport" % fname)
        createNewRow = True
        curRow = None
        createPURow = True
        puRow = None
        for stepdir in stepDirs:
            base  = os.path.basename(stepdir)
            found = stepreg.search(base)
            step  = "Unknown-step"
            if found:
                step = found.groups()[0]
            realstep  = "Unknown-step"
            if "PILEUP" in step:
                found = pureg.search(step)
                if found:
                    realstep = found.groups()[0]
                if createPURow:
                    createPURow = False
                    puRow = Row(table)                
            rootf = os.path.join(stepdir,rootfile)

            if os.path.exists(rootf):
                f = ROOT.TFile(rootf)

                cpu_time_tree = ROOT.TTree()
                f.GetObject("cpu_time_tuple;1",cpu_time_tree)
                if cpu_time_tree:
                    if cpu_time_tree.InheritsFrom("TTree"):
                        data1 = None
                        data2 = None
                        for t in cpu_time_tree:
                            data1 = t.total1
                            data2 = t.total2
                        if data1 and data2:
                            if createNewRow:
                                createNewRow = False
                                curRow = table.newRow(cand)
                            data_tuple = (data1,data2)

                            if "PILEUP" in step:
                                puRow.addEntry(realstep,data_tuple)
                            else:
                                if createNewRow:
                                    createNewRow = False
                                    curRow = table.newRow(cand)

                                curRow.addEntry(step,data_tuple)
                f.Close()
        if puRow == None:
            pass
        else:
            table.addRow(puRow,"%s PILEUP" %cand)                
    return table
                

def createHTMLtab(INDEX,table_dict,ordered_keys,header,caption,name,mode=0):
    cols     = len(ordered_keys)
    totcols  = (cols * 3) + 1
    innercol = 3
    colspan  = totcols - 1
    labels   = []
    if mode == 1:
        labels = ["fs1","fs2","&#x0394;"]
    elif mode == 2 or mode == 3:
        colspan = cols
        innercol = 1
    else:
        labels = ["t1" ,"t2" ,"&#x0394;"]


    INDEX.write("<h3>%s</h3>\n" % header)
    INDEX.write("<table>\n")
    INDEX.write("<caption>%s</caption>\n" % caption)
    INDEX.write("<thead><tr><th></th><th colspan=\"%s\" scope=\"colgroup\">%s</th></tr></thead>" % (colspan,name)) 
    INDEX.write("<tbody>\n")
    for key in ordered_keys:
        INDEX.write("<tr>")
        if key == None:
            INDEX.write("<th></th>")
        else:
            INDEX.write("<td scope=\"row\">")
            INDEX.write(key)
            INDEX.write("</td>")
        for col in table_dict[None]:
            if key == None:
                INDEX.write("<th colspan=\"%s\" scope=\"col\">" % innercol)
                INDEX.write(col)
                INDEX.write("</th>")                            
            else:
                rowdict = table_dict[key].getRowDict()
                if rowdict.has_key(col):
                    if mode == 2:
                        dat = prettySize(rowdict[col])
                        INDEX.write("<td>")
                        INDEX.write("%s" % dat)
                        INDEX.write("</td>")
                        
                    elif mode == 3:
                        dat = rowdict[col]
                        INDEX.write("<td>")
                        INDEX.write("%6.2f" % dat)
                        INDEX.write("</td>")                        
                    else:
                        (data1, data2) = rowdict[col]
                        diff = data2 - data1

                        if mode == 1:
                            diff  = prettySize(diff)
                            data1 = prettySize(data1)
                            data2 = prettySize(data2)
                        
                        seq = [ data1, data2, diff ]
                        for dat in seq:
                            INDEX.write("<td id=\"data\">")
                            if mode == 1:
                                INDEX.write("%s" % dat) # %s if                            
                            else:
                                INDEX.write("%6.2f" % dat) # %s if                            

                            INDEX.write("</td>")
                else:
                    if mode == 2 or mode == 3:
                        INDEX.write("<td>")                                    
                        INDEX.write("N/A")
                        INDEX.write("</td>")                         
                    else:
                        for i in range(3):
                            INDEX.write("<td>")                                    
                            INDEX.write("N/A")
                            INDEX.write("</td>") 
        INDEX.write("</tr>\n")
        # write an additional row if this row is the header row
        # we need to describe the sub columns
        if not (mode == 2 or mode == 3 ):
            if key == None:
                INDEX.write("<tr>")
                INDEX.write("<th>Candles</th>")
                for col in table_dict[None]:
                    INDEX.write("<th>%s</th>" % labels[0])
                    INDEX.write("<th>%s</th>" % labels[1])
                    INDEX.write("<th>%s</th>" % labels[2])
                INDEX.write("</tr>\n")
    INDEX.write("</tbody></table>\n")

    INDEX.write("<br />")    

def stageIgProfReports(remotedir,arch,version):
    '''Publish all IgProf files into one remote directory (new naming convention). Can publish to AFS location or to a local directory on a remote (virtual) machine.'''
    #FIXME: can eliminate this part if using tar pipes... was done with rsynch in mind
    #Compose command to create remote dir:
    if ":" in remotedir: #Remote host local directory case
        (host,dir)=remotedir.split(":")
        mkdir_cmd="ssh %s (mkdir %s;mkdir %s/%s)"%(host,dir,dir,arch)
    else: #AFS or local case
        mkdir_cmd="mkdir %s;mkdir %s/%s"%(remotedir,remotedir,arch)

    #Create remote dir:
    try:
        print mkdir_cmd
        os.system(mkdir_cmd)
        print "Successfully created publication directory"
    except:
        print "Issues with publication directory existence/creation!"
        
    #Copy files over to remote dir
    #replacing rsync with tar pipes since it can hang on AFS (Andreas' experience):
    #rsync_cmd="rsync -avz *_IgProf_*/*.sql3 %s/%s/%s"%(remotedir,arch,version)
    if ":" in remotedir:
        tarpipe_cmd='tar cf - *_IgProf_*/*.sql3 | ssh %s "cd %s/%s; mkdir %s; cd %s; tar xf -; mv *_IgProf_*/*.sql3 .; rmdir *_IgProf_*"'%(host,dir,arch,version,version)
    else:
        tarpipe_cmd='tar cf - *_IgProf_*/*.sql3 | (cd %s/%s; mkdir %s; cd %s; tar xf -; mv *_IgProf_*/*.sql3 .; rmdir *_IgProf_*)'%(remotedir,arch,version,version)
    try:
    #    print rsync_cmd
    #    os.system(rsync_cmd)
        print tarpipe_cmd
        os.system(tarpipe_cmd)
        print "Successfully copied IgProf reports to %s"%remotedir
    except:
        print "Issues with rsyncing to the remote directory %s!"%remotedir

    #Make sure permissions are set for group to be able to write:
    if ":" in remotedir: #Remote host local directory case
        chmod_cmd="ssh %s chmod -R 775 %s/%s"%(host,dir,arch)
    else:
        chmod_cmd="chmod -R 775 %s/%s"%(remotedir,arch)
    try:
        print chmod_cmd
        os.system(chmod_cmd)
        print "Successfully set permissions for IgProf reports directory %s"%remotedir
    except:
        print "(Potential) issues with chmoding the remote directory %s!"%remotedir
    
    return #Later, report here something like the web link to the reports in igprof-navigator...


#####################
#
# Create web report index and create  HTML file for each candle
#
def createWebReports(WebArea,repdir,ExecutionDate,LogFiles,cmsScimarkResults,date,prevrev):

    #Some nomenclature

    Candle = Candles #These need to match the directory names in the work area

    CmsDriverCandle = CandFname #{ #These need to match the cmsDriver.py output filenames


    #Produce a "small" index.html file to navigate the html reports/logs etc
    IndexFile="%s/index.html" % WebArea
    TemplateHtml="%s/doc/index.html" % BASE_PERFORMANCE

    cmsverreg = re.compile("CMSSW_VERSION")
    hostreg   = re.compile("HOST")
    lpathreg  = re.compile("LocalPath")
    fsizereg  = re.compile("FSizeTable")        
    cpureg    = re.compile("CPUTable")    
    proddreg  = re.compile("ProductionDate")
    logfreg   = re.compile("LogfileLinks")
    dirbreg   = re.compile("DirectoryBrowsing")
    pubdreg   = re.compile("PublicationDate")
    candhreg  = re.compile("CandlesHere")
    #Loop line by line to build our index.html based on the template one
    #Copy the perf_style.css file from Validation/Performance/doc

    CandlTmpltHTML="%s/doc/candle.html" % BASE_PERFORMANCE
    if _verbose:
        print "Copying %s/doc/perf_style.css style file to %s/." % (BASE_PERFORMANCE,WebArea)    
        print "Template used: %s" % TemplateHtml

    syscp((BASE_PERFORMANCE + "/doc/perf_style.css"),WebArea + "/.")
    pureg = re.compile("(.*)_PILEUP")    
    try:
        INDEX = open(IndexFile,"w") 
        for NewFileLine in open(TemplateHtml) :
            if cmsverreg.search(NewFileLine):
                if prevrev == "":
                    INDEX.write("Performance Reports for %s\n" % CMSSW_VERSION)
                else:
                    globpath = os.path.join(repdir,"REGRESSION.%s.vs.*" % (prevrev))
                    globs = glob.glob(globpath)
                    if len(globs) < 1:
                        pass
                    else:
                        latestreg = re.compile("REGRESSION.%s.vs.(.*)" % prevrev)
                        found = latestreg.search(os.path.basename(globs[0]))
                        if found:
                            latestrel = found.groups()[0]
                            INDEX.write("Performance Reports with regression: %s VS %s\n" % (prevrev,latestrel))                                                        
                        else:
                            INDEX.write("Performance Reports with regression: %s VS %s\n" % (prevrev,CMSSW_VERSION))                            
            elif hostreg.search(NewFileLine):
                INDEX.write(HOST + "\n")
            #File Size Summary Table
            elif fsizereg.search(NewFileLine):
                #Case of NO-REGRESSION:
                if prevrev == "":
                    fsize_tab = Table()

                    for cand in Candles:
                        fname = CandFname[cand]
                        globpath  = os.path.join(repdir,"%s*_TimeSize" % cand,"%s_*.root" % fname)
                        rootfiles = glob.glob(globpath)
                        rootfiles = map(lambda x: (fname,x), rootfiles)
                        rootfiles.sort(cmp=rootfile_cmp)
                        rootfiles = map(lambda x: x[1], rootfiles)
                        stepreg = re.compile("%s_(.*).root" % fname)
                        createNewRow = True
                        curRow = None
                        createPURow = True
                        puRow = None                        
                        for rootf in rootfiles:
                            base  = os.path.basename(rootf)
                            found = stepreg.search(base)
                            step  = "Unknown-step"
                            if found:
                                step = found.groups()[0]
                            realstep  = "Unknown-step"
                            if "PILEUP" in step:
                                found = pureg.search(step)
                                if found:
                                    realstep = found.groups()[0]
                                if createPURow:
                                    createPURow = False
                                    puRow = Row(fsize_tab)
                            try:
                                statinfo = os.stat(rootf)
                                fsize    = statinfo.st_size
                                if createNewRow:
                                    createNewRow = False
                                    curRow = fsize_tab.newRow(cand)
                                    
                                if "PILEUP" in step:
                                    puRow.addEntry(realstep,fsize)
                                else:
                                    if createNewRow:
                                        createNewRow = False
                                        curRow = fsize_tab.newRow(cand)
                                        
                                    curRow.addEntry(step,fsize)       
                            except IOError, detail:
                                print detail
                            except OSError, detail:
                                print detail
                        if puRow == None:
                            pass
                        else:
                            fsize_tab.addRow(puRow,"%s PILEUP" %cand)                                                                    
                    (ordered_keys,table_dict) = fsize_tab.getTable(1)
                    cols = len(ordered_keys)
                    
                    if len(table_dict) > 1 and cols > 0:
                        createHTMLtab(INDEX,table_dict,ordered_keys,
                                      "Release ROOT file sizes",
                                      "Table showing current release ROOT filesizes in (k/M/G) bytes.",
                                      "Filesizes",2)   
                #Case of REGRESSION vs previous release:
                else:
                    try:
                        globpath = os.path.join(repdir,"REGRESSION.%s.vs.*" % (prevrev))
                        globs = glob.glob(globpath)
                        if len(globs) < 1:
                            raise IOError(2,globpath,"File does not exist","File does not exist")
                        idfile  = open(globs[0])
                        oldpath = ""
                        for line in idfile:
                            oldpath = line
                        oldpath = oldpath.strip()
                        #print "##########TABLE DEBUG :oldpath is %s"%oldpath
                        fsize_tab = Table()

                        for cand in Candles:
                            fname = CandFname[cand]
                            globpath  = os.path.join(repdir,"%s*_TimeSize" % cand,"%s_*.root" % fname)
                            rootfiles = glob.glob(globpath)
                            rootfiles = map(lambda x: (fname,x), rootfiles)
                            rootfiles.sort(cmp=rootfile_cmp)
                            rootfiles = map(lambda x: x[1], rootfiles)
                            stepreg = re.compile("%s_(.*).root" % fname)
                            createNewRow = True
                            curRow = None
                            createPURow = True
                            puRow = None                            
                            for rootf in rootfiles:
                                base  = os.path.basename(rootf)
                                found = stepreg.search(base)
                                step  = "Unknown-step"
                                if found:
                                    step = found.groups()[0]
                                    
                                realstep  = "Unknown-step"
                                if "PILEUP" in step:
                                    found = pureg.search(step)
                                    if found:
                                        realstep = found.groups()[0]
                                    if createPURow:
                                        createPURow = False
                                        puRow = Row(fsize_tab)
                                try:
                                    statinfo = os.stat(rootf)
                                    fsize2   = statinfo.st_size
                                    oldfile  = os.path.join(oldpath,"%s_TimeSize" % cand,base)
                                    if "PILEUP" in step:
                                        oldfile  = os.path.join(oldpath,"%s_PU_TimeSize" % cand,base)
                                    fsize1   = 0

                                    if os.path.exists(oldfile):
                                        statinfo = os.stat(oldfile)
                                        fsize1   = statinfo.st_size
                                    else:
                                        print "######DID NOT FIND Previous file (needed for the filesize table): %s"%oldfile
                                            
                                    if createNewRow:
                                        createNewRow = False
                                        curRow = fsize_tab.newRow(cand)

                                    data_tuple = (fsize1,fsize2)
                                    if "PILEUP" in step:
                                        puRow.addEntry(realstep,data_tuple)
                                    else:
                                        if createNewRow:
                                            createNewRow = False
                                            curRow = fsize_tab.newRow(cand)

                                        curRow.addEntry(step,data_tuple)
                                except IOError, detail:
                                    print detail
                                except OSError, detail:
                                    print detail
                            if puRow == None:
                                pass
                            else:
                                fsize_tab.addRow(puRow,"%s PILEUP" %cand)                                    
                                    
                        (ordered_keys,table_dict) = fsize_tab.getTable()
                        cols = len(ordered_keys)
                    
                        if len(table_dict) > 1 and cols > 0:
                            createHTMLtab(INDEX,table_dict,ordered_keys,
                                          "Release ROOT file sizes",
                                          "Table showing previous release ROOT filesizes, fs1, latest sizes, fs2, and the difference between them &#x0394; in (k/M/G) bytes.",
                                          "Filesizes",1)
                    except IOError, detail:
                        print detail
                    except OSError, detail:
                        print detail
            #CPU Time Summary Table    
            elif cpureg.search(NewFileLine):
                #Case of NO REGRESSION
                if prevrev == "":
                    time_tab = Table()

                    for cand in Candles:
                        fname = CandFname[cand]
                        globpath  = os.path.join(repdir,"%s*_TimeSize" % cand,"%s_*_TimingReport.log" % fname)
                        logfiles = glob.glob(globpath)
                        logfiles = map(lambda x: (fname,x), logfiles)
                        logfiles.sort(cmp=logrep_cmp)                        
                        logfiles = map(lambda x: x[1], logfiles)

                        stepreg = re.compile("%s_(.*)_TimingReport.log" % fname)
                        createNewRow = True
                        curRow = None
                        createPURow = True
                        puRow = None
                        for log in logfiles:
                            base  = os.path.basename(log)
                            found = stepreg.search(base)
                            step  = "Unknown-step"
                            if found:
                                step = found.groups()[0]

                            realstep  = "Unknown-step"
                            if "PILEUP" in step:
                                found = pureg.search(step)
                                if found:
                                    realstep = found.groups()[0]
                                if createPURow:
                                    createPURow = False
                                    puRow = Row(time_tab)
                                
                            data = cpr.getTimingLogData(log)
                            mean = 0
                            i    = 0
                            for evtnum, time in data:
                                mean += time
                                i += 1
                            try:
                                mean = mean / float(i)
                            except ZeroDivisionError, detail:
                                print "WARNING: Could not calculate mean CPU time from log because no events could be parsed", log

                            if "PILEUP" in step:
                                puRow.addEntry(realstep,mean)
                            else:
                                if createNewRow:
                                    createNewRow = False
                                    curRow = time_tab.newRow(cand)

                                curRow.addEntry(step,mean)                                
                        if puRow == None:
                            pass
                        else:
                            time_tab.addRow(puRow,"%s PILEUP" %cand)

                    (ordered_keys,table_dict) = time_tab.getTable(1)
                    cols = len(ordered_keys)
                    
                    if len(table_dict) > 1 and cols > 0:
                        createHTMLtab(INDEX,table_dict,ordered_keys,
                                      "Release CPU times",
                                      "Table showing current release CPU times in secs.",
                                      "CPU Times (s)",3)
                #Case of REGRESSION (CPU Time Summary Table)
                else:


                    ####
                    #
                    # Create the table data structure
                    #
                    cpu_time_tab =  populateFromTupleRoot("cpu_time_tuple",repdir,"timing-regress.root",pureg)
                    

                    ###########
                    #
                    # Create HTML table from table data structure
                    #

                    (ordered_keys,table_dict) = cpu_time_tab.getTable()

                    cols = len(ordered_keys)
                    if len(table_dict) > 1 and cols > 0:
                        createHTMLtab(INDEX,table_dict,ordered_keys,
                                      "Release CPU times",
                                      "Table showing previous release CPU times, t1, latest times, t2, and the difference between them &#x0394; in secs.",
                                      "CPU Times (s)")

                        
                    
            elif lpathreg.search(NewFileLine):
                INDEX.write(repdir + "\n")
            elif proddreg.search(NewFileLine):
                INDEX.write(ExecutionDate + "\n")
            elif logfreg.search(NewFileLine):
                INDEX.write("<br />\n")
                for log in LogFiles:
                    log = os.path.basename(log)
                    if _verbose:
                        print "linking log file %s" % log
                    INDEX.write("<a href=\"./%s\"> %s </a>" % (log,log))
                    INDEX.write("<br />\n")
                #Add the cmsScimark results here:
                INDEX.write("Results for cmsScimark2 benchmark (running on the other cores) available at:\n")
                for cmssci in cmsScimarkResults:
                    scimarkdirs=cmssci.split("/")
                    localdir=scimarkdirs[-2]
                    #print localdir
                    cmssci = os.path.basename(cmssci)
                    relativelink=os.path.join(localdir,cmssci)
                    #print relativelink
                    INDEX.write("<a href=\"%s\"> %s </a>" % (relativelink,cmssci))
                    INDEX.write("<br />\n")


            elif dirbreg.search(NewFileLine):
                #Create a subdirectory DirectoryBrowsing to circumvent the fact the dir is not browsable if there is an index.html in it.
                #Bug! This does not work since it points to index.html automatically!
                #os.symlink("./","%s/DirectoryBrowsing" % WebArea)
                #Actually all the following is done later...
                #Create a physical directory first
                #os.mkdir("%s/DirectoryBrowsing" % WebArea)
                #Then will populate it with symbolic links(once they have been copied there!) from all WebArea files, except index.html, see down towards the end!
                INDEX.write("Click <a href=\"./DirectoryBrowsing/\">here</a> to browse the directory containing all results (except the root files)\n")

            elif pubdreg.search(NewFileLine):
                INDEX.write(date + "\n")
            elif candhreg.search(NewFileLine):
                for acandle in Candle:
                    globpath = os.path.join(repdir,"%s_*" % acandle)
                    globs = glob.glob(globpath)
                    if len(globs) > 0:
                        candlHTML = "%s.html" % acandle
                        INDEX.write("<a href=\"./%s\"> %s </a>" % (candlHTML,acandle))
                        INDEX.write("<br />\n")
                    
                        candlHTML=os.path.join(WebArea,candlHTML)
                        createCandlHTML(CandlTmpltHTML,candlHTML,acandle,WebArea,repdir,ExecutionDate,LogFiles,cmsScimarkResults,date,prevrev)
            else:
                INDEX.write(NewFileLine)

        #End of while loop on template html file
        INDEX.close()
    except IOError, detail:
        print "Error: Could not create index Html file for some reason, check position. Details : %s" % detail

########################
#
# Grab dirs that end in strings defined in DirName
#
def getDirnameDirs(repdir,WebArea):
    Dir = os.listdir(repdir)
    def _containsDirName(elem):
        return reduce(lambda x,y: x or y,map(lambda x: x in elem, DirName))
    def _print4Lambda(elem,WebArea):
        if _verbose:
            print "Copying %s to %s\n" %  (elem,WebArea)

    dirstocp = filter(lambda x: _containsDirName(x),map(lambda x: repdir + x,Dir))
    map(lambda x: _print4Lambda(x,WebArea),dirstocp)
    syscp(dirstocp,WebArea + "/")
    os.mkdir("%s/DirectoryBrowsing" % WebArea)
    for file in os.listdir(WebArea):
        if file != "index.html": #Potential maintenance issue if the index.html changes name to something the server automatically displays when pointing to the directory...
            #Use relative path ".." instead of WebArea to avoid problems when copying stuff to a remote server!
            os.symlink("%s/%s"%("..",file),"%s/DirectoryBrowsing/%s" % (WebArea,file))

#######################
#
# Upload stage to remote location
def syncToRemoteLoc(stage,drive,path,port):
    stage = addtrailingslash(stage)
    cmd = "rsync -avz"
    # We must, MUST, do os.path.normpath otherwise rsync will dump the files in the directory
    # we specify on the remote server, rather than creating the CMS_VERSION directory
    #--rsh=\"ssh -l relval\" 
    args = "--port=%s %s %s:%s" % (port,os.path.normpath(stage),drive,path)
    retval = -1
    if _dryrun:
        print              cmd + " --dry-run " + args 
        retval = os.system(cmd + " --dry-run " + args )
    else:
        print cmd+" "+args
        retval = os.system(cmd + " " + args)
    return retval

################
# 
# Delete tmp dir if we used it
def delTmpDir():
    if os.path.exists(TMP_DIR) and IS_TMP:
        os.system("rm -Rf " + TMP_DIR)

#####################
#
# Some functions used for copying

def getRelativeDir(parent,child,keepTop=True):
    def _walkpath(path):
        dirs = []
        while True:
            head , tail = os.path.split(path)
            if tail == "":
                break
            dirs.append(tail)
            path = head
        for i in range(len(dirs)-1,-1,-1):
            adir = dirs[i]
            yield adir
        return
    pwalk = _walkpath(parent)
    n = 0
    try:
        while True:
            pwalk.next()
            n += 1
    except StopIteration:
        pass

    if keepTop:
        n = n - 1

    cwalk = _walkpath(child)
    try:
        #prewalk
        for x in range(n):
            cwalk.next()
    except StopIteration:
        print "ERROR: Unable to determine relative dir"
        raise ReldirExcept

    relpath = ""
    try:
        while True:
            relpath=os.path.join(relpath,cwalk.next())
    except StopIteration:
        pass
    return relpath

def docopy(src,dest):
    try:
        copy2(src,dest)
    except OSError, detail:
        print "WARNING: Could not copy %s to %s because %s" % (src,dest,detail)        
    except IOError, detail:
        print "WARNING: Could not copy %s to %s because %s" % (src,dest,detail)
    else:
        if _verbose:
            print "Copied %s to %s" % (src,dest)

def copytree4(src,dest,keepTop=True):
    def _getNewLocation(source,child,dst,keepTop=keepTop):
        place = getRelativeDir(source,child,keepTop=keepTop)
        return os.path.join(dst,place)
    def _copyFilter(source,dst,curdir,fsnodes,filter,dirs=False):
        for node in fsnodes:
            dontFilter = True
            filterExist = not len(filter) == 0
            if filterExist:
                dontFilter = not reduce(lambda x,y: x or y,map(lambda x: fnmatch.fnmatch(node,x),filter))
            if dontFilter:
                node = os.path.join(curdir,node) # convert to absolute path
                try:
                    newnode = _getNewLocation(source,node,dst)
                    if dirs:
                        os.mkdir(newnode)                
                    else:
                        copy2(node,newnode)
                except IOError, detail:
                    print "WARNING: Could not copy %s to %s because %s" % (node,newnode,detail)
                except OSError, detail:
                    print "WARNING: Could not copy %s to %s because %s" % (src,dest,detail)                    
                except ReldirExcept:
                    print "WARNING: Could not determine new location for source %s into destination %s" % (source,dst)
                else:
                    if len(filter) > 0:
                        try:
                            match = fnmatch.fnmatch(node,filter[0])
                            assert not match
                        except AssertionError, detail:
                            print node, filter[0], match
                            raise RuntimeError
                    if _verbose:
                        if "root" in node:                            
                            print "Filter %s Copied %s to %s" % (dontFilter,node,newnode)
                            print "fnmatch %s" % fnmatch.fnmatch(node,cpFileFilter[0]) 
                    
    gen  = os.walk(src)
    try:
        newloc = _getNewLocation(src,src,dest)

        os.mkdir(newloc)
        try:
            while True:
                step   = gen.next()
                curdir = step[0]
                dirs   = step[1]
                files  = step[2]

                _copyFilter(src,dest,curdir,dirs,cpDirFilter,dirs=True)
                _copyFilter(src,dest,curdir,files,cpFileFilter)

        except StopIteration:
            pass        
    except IOError, detail:
        print "WARNING: Could not copy %s to %s because %s" % (src,dest,detail)
    except OSError, detail:
        print "WARNING: Could not copy %s to %s because %s" % (src,dest,detail)        
    except ReldirExcept:
        print "WARNING: Could not determine the new location for source %s into destination %s" % (src,dest)
        
def syscp(srcs,dest):
    if type(srcs) == type(""):
        if os.path.exists(srcs):
            if os.path.isdir(srcs):
                copytree4(srcs,dest)
            else:
                docopy(srcs,dest)
        else:
            print "ERROR: file to be copied %s does not exist" % foo            
    else:
        for src in srcs:
            if os.path.exists(src):
                if os.path.isdir(src):
                #copy tree
                    copytree4(src,dest)
                else:
                    docopy(src,dest)
            else:
                print "ERROR: file to be copied %s does not exist" % foo
            
def print_header():
    print "%s\n" % PROG_NAME

if __name__ == "__main__":
    main()
