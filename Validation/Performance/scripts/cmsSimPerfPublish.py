#!/usr/bin/python
#G.Benelli Jan 22 2007, J. Nicolson 12 Aug 2008
#A little script to move Simulation Performance Suite
#relevant html and log files into our public area
#/afs/cern.ch/cms/sdt/web/performance/simulation/
#Set here the standard number of events (could become an option... or could be read from the log...)

import dircache as dc
import tempfile as tmp
import optparse as opt
import re, os, sys, time, glob, socket
from stat import *


def setDefaultPaths():
    global DEF_RELVAL, DEF_SIMUL
    DEF_RELVAL = "/afs/cern.ch/cms/sdt/web/performance/RelVal/%s"     % CMSSW_VERSION
    DEF_SIMUL  = "/afs/cern.ch/cms/sdt/web/performance/simulation/%s" % CMSSW_VERSION

TimeSizeNumOfEvents = 100
IgProfNumOfEvents   = 5
ValgrindNumOfEvents = 1

DirName=( #These need to match the candle directory names ending (depending on the type of profiling)
          "TimeSize",
          "IgProf",
          "Valgrind"
          )

#This hash and the code that checked for this is obsolete: it was for backward compatibility with cfg version
#of the suite (need to clean it up sometime)
#StepLowCaps={
#              "SIM"  : "sim",
#              "DIGI" : "digi",
#              "RECO" : "reco",
#              "DIGI_PILEUP" : "digi_pileup",
#              "RECO_PILEUP" : "reco_pileup",
#              }

class CopyError(Exception):
    'Copy error: Source or destination does not exist'

def getDate():
    return time.ctime()

def getcmdBasic(cmd):
    return os.popen4(cmd)[1].read().strip()

def getcmd(command):
    if _debug:
        print command
    return os.popen4(command)[1].read().strip()

def syscp(src,dest,opts=""):
    src  = os.path.normpath(src)
    dest = os.path.normpath(dest)
    try:
        srcExists  = reduce(lambda x,y : x and y, map(os.path.exists,glob.glob(src )))
        destExists = reduce(lambda x,y : x and y, map(os.path.exists,glob.glob(dest)))
    except TypeError:
        print "ERROR: Failed to find some files. Are you sure you are in the report dir or have specified it correctly? Otherwise this might be a bug."
        print "exiting..."
        sys.exit()
    if (not srcExists):
        print "Error: %s src does not exist" % src
        raise CopyError    
    if (not destExists):
        print "Error: %s destination does not exist" % dest 
        raise CopyError
    return os.system("cp %s %s %s" % (opts,src,dest))

def optionparse():
    global PROG_NAME, _debug, _dryrun
    PROG_NAME   = os.path.basename(sys.argv[0])

    parser = opt.OptionParser(usage=("""%s [HOST:]DEST_PATH [Options]

    Arguments:
        WEB_AREA - local, relval, covms or ...

    Examples:  
       Perform 
        ./%s 
       Perform 
        ./%s 
       Perform 
        ./%s 
       Perform 
        ./%s """
      % ( PROG_NAME, PROG_NAME,PROG_NAME,PROG_NAME,PROG_NAME)))
    
    devel  = opt.OptionGroup(parser, "Developer Options",
                                     "Caution: use these options at your own risk."
                                     "It is believed that some of them bite.\n")
    #parser.set_defaults(debug=False)

    parser.add_option(
        '--relval',
        action="store_true",
        dest='relval',
        help='Use the default simulation location',
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
        '-p',
        '--port',
        type='int',
        dest='port',
        help='Use a particular port number to rsync material to a remote server',
        metavar='<PORT>'
        )

    devel.add_option(
        '-d',
        '--debug',
        action="store_true",
        dest='debug',
        help='Show debug output',
        #metavar='DEBUG',
        )

    devel.add_option(
        '--dry-run',
        action="store_true",
        dest='dryrun',
        help='dry-run output remote sync commands that will be run but do staging',
        #metavar='DEBUG',
        )

    parser.set_defaults(debug=False,simulation=False,relval=False,port=873,pretend=False)
    parser.add_option_group(devel)

    (options, args) = parser.parse_args()

    _debug = options.debug
    _dryrun = options.dryrun

    numofargs = len(args) 

    if (options.simulation and options.relval) or ((options.simulation or options.relval) and numofargs >= 1):
        parser.error("You can not specify simulation and relval together. Neither can you specify simulation or relval AND a path")
        sys.exit()

    return (options, args)

def get_environ():
    global CMSSW_VERSION, CMSSW_RELEASE_BASE, CMSSW_BASE, HOST, USER, BASE_PERFORMANCE, CMSSW_WORK
    try:
        CMSSW_VERSION=os.environ['CMSSW_VERSION']
        CMSSW_RELEASE_BASE=os.environ['CMSSW_RELEASE_BASE']
        CMSSW_BASE=os.environ['CMSSW_BASE']
        HOST=os.environ['HOST']
        USER=os.environ['USER']
        CMSSW_WORK = os.path.join(CMSSW_BASE,"work")
    except KeyError:
        print "ERROR: Could not retrieve some necessary environment variables. Have you ran scramv1 runtime -csh yet?"
        sys.exit()

    LocalPath=getcmdBasic("pwd")
    ShowTagsResult=getcmdBasic("showtags -r")

    #Adding a check for a local version of the packages
    PerformancePkg="%s/src/Validation/Performance"                   % CMSSW_BASE
    if (os.path.exists(PerformancePkg)):
        BASE_PERFORMANCE=PerformancePkg
        print "**[cmsSimPerfPublish.pl]Using LOCAL version of Validation/Performance instead of the RELEASE version**"
    else:
        BASE_PERFORMANCE="%s/src/Validation/Performance"             % CMS_RELEASE_BASE

    return (LocalPath,ShowTagsResult)

def getStagingArea(options,args):
    global TMP_DIR
    numofargs = len(args) 

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

    ####
    #
    # Determine if location is remote
    #
    # Try not to re-arrange we don't want to assume that default locations are not remote
    #
    ####
    #atreg = re.compile("\[^\\\]+@")
    drive, path = uri.split(":",1)
    if drive == "":
        path = os.path.normpath(path)
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

    if (not remote) and (not options.port == 873) :
        print "WARNING: Can not use a port if not performing a remote copy, ignoring"
    port = options.port

    ###
    #
    # Determine Staging Area
    #

    StagingArea=""
    TMP_DIR = ""
    localExists = os.path.exists("%s/%s" % (CMSSW_WORK,CMSSW_VERSION))
    
    if remote:
        TMP_DIR=tmp.mkdtemp(prefix="/tmp/%s" % PROG_NAME)
        StagingArea = TMP_DIR
    #Local but dir already exists
    elif defaultlocal and localExists:
        TMP_DIR=tmp.mkdtemp(prefix="%s/%s" % (CMSSW_WORK,CMSSW_VERSION))
        StagingArea = TMP_DIR
        print "WARNING: %s already exists, creating a temporary staging area %s" % (CMSSW_WORK,TMP_DIR)
    #Local cases
    elif defaultlocal:
        StagingArea = CMSSW_WORK
        print "**User did not specify location of results, staging in default %s**" % StagingArea 
    else:
        print "**User chose to publish results in a local directory**" 
        StagingArea = path
        if not os.path.exists(path):
            os.system("mkdir -p %s" % path)

    ######
    #
    # create Version dir
    StagingArea="%s/%s" % (StagingArea,CMSSW_VERSION)
    os.system("mkdir -p %s" % StagingArea)

    return (drive,path,remote,StagingArea,port)

def isStagingAreaEmpty(stage):
    afsreg   = re.compile("^\..*afs.*")
    Contents = filter(not afsreg.search,dc.opendir(stage)) 
   
    if (len(Contents) == 0):
        print "The area %s is ready to be populated!" % stage
        return True
    else:
        print "The area %s is not empty (don't worry about .afs files we filter them)!" % stage
        return False

def scanReportArea(repdir):
    date=getDate()
    LogFiles  = glob.glob(repdir + "cms*.log") 
    print "Found the following log files:"
    print LogFiles

    cmsScimarkDir = glob.glob(repdir + "cmsScimarkResults_*")
    print "Found the following cmsScimark2 results directories:"
    print cmsScimarkDir

    #htmlreg = re.compile(".*\.html")
    cmsScimarkResults = []
    for dir in cmsScimarkDir:
        htmlfiles = glob.glob(dir + "/*.html") #filter(htmlreg.search,dc.listdir(dir))
        #htmlfiles = map(lambda x : dir + "/" + x,htmlfiles)
        map(cmsScimarkResults.append,htmlfiles)

    ExecutionDate = ""
    ExecutionDateSec=0
    cmsreg = re.compile("^cmsCreateSimPerfTest")
    for logf in LogFiles:
        if cmsreg.search(logf):
            ExecutionDateLastSec = os.stat(logf)[ST_CTIME]
            ExecutionDateLast    = os.stat(logf)[ST_MTIME]
            print "Execution (completion) date for %s was: %s" % (logf,ExecutionDateLast)
            if (ExecutionDateLastSec > ExecutionDateSec):
                ExecutionDateSec = ExecutionDateLastSec
                ExecutionDate    = ExecutionDateLast

    return (ExecutionDate,LogFiles,date,cmsScimarkResults)

def copyReportsToStaging(repdir,stage):
    print "Copying the logfiles to %s/." % stage
    syscp(repdir + "cms*.log",stage + "/.","-pR")
    print "Copying the cmsScimark2 results to the %s/." % stage
    syscp(repdir + "cmsScimarkResults_*",stage + "/.","-pR")

def createLogFile(LogFile,date,LocalPath,ShowTagsResult):
    try:
        LOG = open(LogFile,"w") 
        print "Writing Production Host, Location, Release and Tags information in %s" % LogFile 
        LOG.write("These performance tests were executed on host %s and published on %s" % (HOST,date))
        LOG.write("They were run in %s" % LocalPath)
        LOG.write("Results of showtags -r in the local release:\n%s" % ShowTagsResult)
        LOG.close()
    except IOError:
        print "Could not correct create the log file for some reason"

def getProfileReportLink(CurrentCandle,CurDir,step,CurrentProfile,Profiler):

    ProfileTemplate="%s_%s/*_%s_%s*/%s" % (CurrentCandle,CurDir,step,CurrentProfile,Profiler)
    #There was the issue of SIM vs sim (same for DIGI) between the previous RelVal based performance suite and the current.
    ProfileTemplateLowCaps="%s_%s/*_%s_%s*/%s" % (CurrentCandle,CurDir,step.lower(),CurrentProfile,Profiler)
    ProfileReportLink=getcmd("ls %s 2>/dev/null" % ProfileTemplate)
    if ( CurrentCandle not in ProfileReportLink) : #no match with caps try low caps
        ProfileReportLink=getcmd("ls %s 2>/dev/null" % ProfileTemplateLowCaps)
    return ProfileReportLink

def writeReportLink(INDEX,ProfileReportLink,CurrentProfile,step,NumOfEvents,Profiler=""):
    if Profiler == "":
        INDEX.write("<li><a href=\"%s\">%s %s (%s events)</a></li>\n" % (ProfileReportLink,CurrentProfile,step,NumOfEvents))
    else:
        INDEX.write("<li><a href=\"%s\">%s %s %s (%s events)</a></li>\n" % (ProfileReportLink,CurrentProfile,Profiler,step,NumOfEvents))

def createCandlHTML(tmplfile,candlHTML,CurrentCandle,WebArea,repdir,ExecutionDate,LogFiles,cmsScimarkResults,date):
    NumOfEvents={ #These numbers are used in the index.html they are not automatically matched to the actual
                   #ones (one should automate this, by looking into the cmsCreateSimPerfTestPyRelVal.log logfile)
                DirName[0] : TimeSizeNumOfEvents,
                DirName[1] : IgProfNumOfEvents,
                DirName[2] : ValgrindNumOfEvents
                }

    Profile=( #These need to match the profile directory names ending within the candle directories
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
    OutputHtml={ #These are the filenames to be linked in the index.html page for each profile
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
    Step=(
           "GEN,SIM",
           "DIGI",
           "L1",
           "DIGI2RAW",
           "HLT",
           "RAW2DIGI",
           "RECO",
           "DIGI_PILEUP",
           "L1_PILEUP",
           "DIGI2RAW_PILEUP",
           "HLT_PILEUP",
           "RAW2DIGI_PILEUP",
           "RECO_PILEUP"
           )

    CAND = open(candlHTML,"w")

    candnreg  = re.compile("CandleName")
    candhreg  = re.compile("CandlesHere")

    for line in open(tmplfile):
        if candhreg.search(line):
            CAND.write("<table cellpadding=\"20px\" border=\"1\"><tr><td>\n")
            CAND.write("<h2>")
            CAND.write(CurrentCandle)
            CAND.write("</h2>\n")
            CAND.write("<div style=\"font-size: 13\"> \n")
            for CurDir in DirName:

                LocalPath = "%s%s_%s" % (repdir,CurrentCandle,CurDir)
                CandleLogFiles = getcmd("sh -c 'find %s -name \"*.log\" 2> /dev/null'" % LocalPath)
                CandleLogFiles = filter("".__ne__,CandleLogFiles.strip().split("\n"))


                if (len(CandleLogFiles)>0):
                    CAND.write("<p><strong>Logfiles for %s</strong></p>\n" % CurDir)
                    for cand in CandleLogFiles:
                        print "Found %s in %s\n" % (cand,LocalPath)
                        syscp(cand,WebArea + "/.","-pR")
                        CAND.write("<a href=\"%s\">%s </a>" % (cand,cand))
                        CAND.write("<br />\n")

                PrintedOnce = False
                for CurrentProfile in Profile:

                    for step in Step :

                        ProfileReportLink = getProfileReportLink(CurrentCandle,
                                                                 CurDir,
                                                                 step,
                                                                 CurrentProfile,
                                                                 OutputHtml[CurrentProfile])

                        if (CurrentProfile in ProfileReportLink):
                            #It could also not be there

                            if (PrintedOnce==False): 
                                #Making sure it's printed only once per directory (TimeSize, IgProf, Valgrind) each can have multiple profiles

                                #This is the "title" of a series of profiles, (TimeSize, IgProf, Valgrind)
                                CAND.write("<p><strong>%s</strong></p>\n" % CurDir)
                                CAND.write("<ul>\n")
                                PrintedOnce=True
                            #Special cases first (IgProf MemAnalyse and Valgrind MemCheck)
                            if (CurrentProfile == Profile[7]):
                                for i in range(0,3,1):
                                    ProfileReportLink = getProfileReportLink(CurrentCandle,
                                                                             CurDir,
                                                                             step,
                                                                             CurrentProfile,
                                                                             IgProfMemAnalyseOut[i])
                                    if (CurrentProfile in ProfileReportLink ) :#It could also not be there
                                        writeReportLink(CAND,ProfileReportLink,CurrentProfile,step,NumOfEvents[CurDir],Profiler=IgProfMemAnalyseOut[i])


                            elif (CurrentProfile == Profile[9]):

                                for i in range(0,3,1):
                                    ProfileReportLink = getProfileReportLink(CurrentCandle,
                                                                             CurDir,
                                                                             step,
                                                                             CurrentProfile,
                                                                             memcheck_valgrindOut[i])
                                    if (CurrentProfile in ProfileReportLink) : #It could also not be there
                                        writeReportLink(CAND,ProfileReportLink,CurrentProfile,step,NumOfEvents[CurDir],Profiler=memcheck_valgrindOut[i])

                            else:
                                writeReportLink(CAND,ProfileReportLink,CurrentProfile,step,NumOfEvents[CurDir])


                if PrintedOnce:
                    CAND.write("</ul>\n")
                PrintedOnce=False
            CAND.write("</div>\n")
            CAND.write("<hr />")
            CAND.write("<br />\n")
            CAND.write("</td></tr></table>\n")
        elif candnreg.search(line):
            CAND.write(CurrentCandle)
        else:
            CAND.write(line)
            
    CAND.close()            

def createHTMLidx(WebArea,repdir,ExecutionDate,LogFiles,cmsScimarkResults,date):

    #Some nomenclature



    Candle=( #These need to match the directory names in the work area
        "HiggsZZ4LM200",
        "MinBias",
        "SingleElectronE1000",
        "SingleMuMinusPt10",
        "SinglePiMinusE1000",
        "TTbar",
        "QCD_80_120"
        )
    CmsDriverCandle={ #These need to match the cmsDriver.py output filenames
        Candle[0] : "HZZLLLL_200",
        Candle[1] : "MINBIAS",
        Candle[2] : "E_1000",
        Candle[3] : "MU-_pt_10",
        Candle[4] : "PI-_1000",
        Candle[5] : "TTBAR",
        Candle[6] : "QCD_80_120"
        }

    #Produce a "small" index.html file to navigate the html reports/logs etc
    IndexFile="%s/index.html" % WebArea
    INDEX = open(IndexFile,"w") 
    print "Writing an index.html file with links to the profiles report information for easier navigation\n"
    TemplateHtml="%s/doc/index.html" % BASE_PERFORMANCE
    print "Template used: %s" % TemplateHtml

    cmsverreg = re.compile("CMSSW_VERSION")
    hostreg   = re.compile("HOST")
    lpathreg  = re.compile("LocalPath")
    proddreg  = re.compile("ProductionDate")
    logfreg   = re.compile("LogfileLinks")
    dirbreg   = re.compile("DirectoryBrowsing")
    pubdreg   = re.compile("PublicationDate")
    candhreg  = re.compile("CandlesHere")
    #Loop line by line to build our index.html based on the template one
    #TEMPLATE =  #||die "Couldn't open file $TemplateHtml - $!\n"

    #Copy the perf_style.css file from Validation/Performance/doc
    print "Copying %s/doc/perf_style.css style file to %s/." % (BASE_PERFORMANCE,WebArea)


    CandlTmpltHTML="%s/doc/candle.html" % BASE_PERFORMANCE
    syscp(BASE_PERFORMANCE + "/doc/perf_style.css",WebArea + "/.","-pR")

    for NewFileLine in open(TemplateHtml) :
        if cmsverreg.search(NewFileLine):
            INDEX.write(CMSSW_VERSION + "\n")
        elif hostreg.search(NewFileLine):
            INDEX.write(HOST + "\n")
        elif lpathreg.search(NewFileLine):
            INDEX.write(repdir + "\n")
        elif proddreg.search(NewFileLine):
            INDEX.write(ExecutionDate + "\n")
        elif logfreg.search(NewFileLine):
            INDEX.write("<br />\n")
            for log in LogFiles:
                INDEX.write("<a href=\"%s\"> %s </a>" % (log,log))
                INDEX.write("<br /><br />\n")
            #Add the cmsScimark results here:
            INDEX.write("Results for cmsScimark2 benchmark (running on the other cores) available at:\n")
            INDEX.write("<br /><br />\n")
            for cmssci in cmsScimarkResults:
                INDEX.write("<a href=\"%s\"> %s </a>" % (cmssci,cmssci))
                INDEX.write("<br /><br />\n")
 

        elif dirbreg.search(NewFileLine):
            #Create a subdirectory DirectoryBrowsing to circumvent the fact the dir is not browsable if there is an index.html in it.
            os.system("mkdir %s/DirectoryBrowsing" % WebArea)
            INDEX.write("Click <a href=\"./DirectoryBrowsing/\">here</a> to browse the directory containing all results (except the root files)\n")

        elif pubdreg.search(NewFileLine):
            INDEX.write(date + "\n")
        elif candhreg.search(NewFileLine):
            for acandle in Candle:
                candlHTML = "%s.html" % acandle
                INDEX.write("<a href=\"./%s\"> %s </a>" % (candlHTML,acandle))
                INDEX.write("<br /><br />\n")
                candlHTML="%s/%s" % (WebArea,candlHTML)
                createCandlHTML(CandlTmpltHTML,candlHTML,acandle,WebArea,repdir,ExecutionDate,LogFiles,cmsScimarkResults,date)
        else:
            INDEX.write(NewFileLine)

    #End of while loop on template html file
    INDEX.close()

def filterStage(repdir,WebArea):
    Dir = os.listdir(repdir)
    for CurrentDir in Dir:
        CurrentDir = repdir + CurrentDir
        for dirn in DirName:
            if (dirn in CurrentDir): # get rid of possible spurious dirs
                print "Copying %s to %s\n"            % (CurrentDir,WebArea)        
                CopyDir=syscp(CurrentDir,WebArea + "/.","-pR")
                RemoteDirRootFiles="%s/%s/*.root"     % (WebArea,CurrentDir) 
                RemoveRootFiles=os.system("rm -Rf %s" % RemoteDirRootFiles)

def createSymLink(WebArea):
    dc.reset()
    DirectoryContent=dc.listdir(WebArea)
    for content in DirectoryContent:
        if ((not (content == "index.html")) and (not(content == "DirectoryBrowsing"))):
            os.system("ln -s %s/%s %s/DirectoryBrowsing/%s" % (WebArea,content,WebArea,content))

def syncToRemoteLoc(stage,drive,path,port):
    bsreg = re.compile("/$")
    if os.path.isdir(stage) and bsreg.search(stage):
        bsreg.sub(r"", stage)
    cmd = "rsync --port=%s %s %s:%s" % (port,stage,drive,path)
    if _dryrun:
        print cmd
    else:
        os.system(cmd)
    return False

def delTmpDir():
    os.system("rm -Rf " + TMP_DIR)

def main():

    #Get  environment variables
    (LocalPath, ShowTagsResult) = get_environ()

    setDefaultPaths()

    (options,args) = optionparse()

    repdir = os.path.normpath(".") + "/" # the report dir, may want to change this or add option in future

    (drive,path,remote,stage,port) = getStagingArea(options,args)

    #if not isStagingAreaEmpty(stage):
    #    sys.exit()

    (ExecutionDate,LogFiles,date,cmsScimarkResults) = scanReportArea(repdir)

    copyReportsToStaging(repdir,stage)

    #Produce a small logfile with basic info on the Production area
    createLogFile("%s/ProductionLog.txt" % stage,date,repdir,ShowTagsResult)

    try:
        INDEX = createHTMLidx(stage,
                              repdir,
                              ExecutionDate,
                              LogFiles,
                              cmsScimarkResults,
                              date)

    except IOError:
        print "Error: Could not create index Html file for some reason, check position"
        sys.exit()

    filterStage(repdir,stage)

    if remote:
        syncToRemoteLoc(stage,drive,path,port)
        delTmpDir()
    #elif defaultlocal:
    #    #Creating symbolic links to the web area in subdirectory to allow directory browsing:
    #    createSymLink(WebArea)

    #if isTemp:
    #    os.system("rm -Rf " + os.path.dirname(WebArea))

if __name__ == "__main__":
    main()
