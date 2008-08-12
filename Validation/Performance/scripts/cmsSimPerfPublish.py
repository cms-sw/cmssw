#!/usr/bin/python
#G.Benelli Jan 22 2007, J. Nicolson 12 Aug 2008
#A little script to move Simulation Performance Suite
#relevant html and log files into our public area
#/afs/cern.ch/cms/sdt/web/performance/simulation/
#Set here the standard number of events (could become an option... or could be read from the log...)

import dircache as dc
import tempfile as tmp
import optparse as opt
import re, os, sys, time, glob
from stat import *

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

def getcmd(command):
    if _debug:
        print command
    return os.popen4(command)[1].read().strip()

def syscp(src,dest,opts=""):
    src  = os.path.normpath(src)
    dest = os.path.normpath(dest)
    srcExists  = reduce(lambda x,y : x and y, map(os.path.exists,glob.glob(src )))
    destExists = reduce(lambda x,y : x and y, map(os.path.exists,glob.glob(dest)))
    if (not srcExists):
        print "Error: %s src does not exist" % src
        raise CopyError    
    if (not destExists):
        print "Error: %s destination does not exist" % dest 
        raise CopyError
    return os.system("cp %s %s %s" % (opts,src,dest))

def optionparse():
    global PROG_NAME
    global _debug
    global UserWebArea
    PROG_NAME   = os.path.basename(sys.argv[0])

    parser = opt.OptionParser(usage=("""%s WEB_AREA [Options]

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

    #parser.add_option(
    #    '-u',
    #    '--usersteps',
    #    type='string',
    #    dest='userSteps',
    #    help='Which steps to run',
    #    metavar='<STEPS>',
    #    )

    devel.add_option(
        '-d',
        '--debug',
        action="store_true",
        dest='debug',
        help='Show debug output',
        #metavar='DEBUG',
        )

    parser.set_defaults(debug=False)
    parser.add_option_group(devel)

    (options, args) = parser.parse_args()

    _debug = options.debug

    numofargs = len(args) 

    if not numofargs == 1:
        parser.error("There are not enough arguments specified to run this program."
                     " Please determine the correct arguments from the usage information above."
                     " Run with --help option for more information.")
        sys.exit()

    UserWebArea = args[0]

    return (options, args)

def get_environ():
    try:
        CMSSW_VERSION=os.environ['CMSSW_VERSION']
        CMSSW_RELEASE_BASE=os.environ['CMSSW_RELEASE_BASE']
        CMSSW_BASE=os.environ['CMSSW_BASE']
        HOST=os.environ['HOST']
        USER=os.environ['USER']
    except KeyError:
        print "ERROR: Could not retrieve some necessary environment variables. Have you ran scramv1 runtime -csh yet?"
        sys.exit()

    LocalPath=getcmd("pwd")
    ShowTagsResult=getcmd("showtags -r")

    #Adding a check for a local version of the packages
    PerformancePkg="%s/src/Validation/Performance"                   % CMSSW_BASE
    if (os.path.exists(PerformancePkg)):
        BASE_PERFORMANCE=PerformancePkg
        print "**[cmsSimPerfPublish.pl]Using LOCAL version of Validation/Performance instead of the RELEASE version**"
    else:
        BASE_PERFORMANCE="%s/src/Validation/Performance"             % CMS_RELEASE_BASE

    return (CMSSW_VERSION,CMSSW_RELEASE_BASE,CMSSW_BASE,HOST,USER,BASE_PERFORMANCE,LocalPath,ShowTagsResult)

def getWebArea(CMSSW_VERSION,UserWebArea):
    isTemp = 0
    #Define the web publishing area
    if (UserWebArea == "simulation"): 
        WebArea="/afs/cern.ch/cms/sdt/web/performance/simulation/%s" % CMSSW_VERSION
        print "Publication web area: %s" % WebArea
    elif (UserWebArea == "relval"):
        WebArea="/afs/cern.ch/cms/sdt/web/performance/RelVal/%s"     % CMSSW_VERSION
        print "Publication web area: %s" % WebArea
    elif (UserWebArea == "local"):
        isTemp = 1
        WebArea=tmp.mkdtemp(prefix="/tmp/%s" % PROG_NAME)
        WebArea="%s/%s" % (WebArea,CMSSW_VERSION)
        #WebArea="/tmp/%s/%s" % (USER,CMSSW_VERSION);
        print "**User chose to publish results in a local directory**"
        print "Creating local directory %s" % WebArea
        os.system("mkdir %s" % WebArea)
    else:
        print "No publication directory specified!\nPlease choose between simulation, relval or local\nE.g.: cmsSimPerfPublish.pl local"
        sys.exit()

    return (WebArea,isTemp)

def isEmptyWebArea(WebArea,isTemp):
    afsreg   = re.compile("^\..*afs.*")
    Contents = filter(not afsreg.search,dc.opendir(WebArea)) 
   
    if (len(Contents) == 0):
        print "The area %s is ready to be populated!" % WebArea
        return 1
    else:
        print "The area %s is not empty (don't worry about .afs files we filter them)!" % WebArea
        if (isTemp):
            os.system("rm -Rf " + WebArea)
        return 0

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

def copyReportsToWeb(WebArea):
    print "Copying the logfiles to %s/." % WebArea
    syscp("./cms*.log",WebArea + "/.","-pR")
    print "Copying the cmsScimark2 results to the %s/." % WebArea
    syscp("./cmsScimarkResults_*",WebArea + "/.","-pR")

def createLogFile(LogFile,HOST,date,LocalPath,ShowTagsResult):
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

def createHTMLidx(WebArea,BASE_PERFORMANCE,CMSSW_VERSION,HOST,LocalPath,ExecutionDate,LogFiles,cmsScimarkResults,date):

    #Some nomenclature

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

    syscp(BASE_PERFORMANCE + "/doc/perf_style.css",WebArea + "/.","-pR")

    for NewFileLine in open(TemplateHtml) :
        if cmsverreg.search(NewFileLine):
            INDEX.write(CMSSW_VERSION + "\n")
        elif hostreg.search(NewFileLine):
            INDEX.write(HOST + "\n")
        elif lpathreg.search(NewFileLine):
            INDEX.write(LocalPath + "\n")
        elif proddreg.search(NewFileLine):
            INDEX.write(ExecutionDate + "\n")
        elif logfreg.search(NewFileLine):
            INDEX.write("<br />\n")
            for log in LogFiles:
                #chomp($_)
                #$LogFileLink="$WebArea/"."$_";
                INDEX.write("<a href=\"%s\"> %s </a>" % (log,log))
                INDEX.write("<br /><br />\n")
            #Add the cmsScimark results here:
            INDEX.write("Results for cmsScimark2 benchmark (running on the other cores) available at:\n")
            INDEX.write("<br /><br />\n")
            for cmssci in cmsScimarkResults:
                INDEX.write("<a href=\"%s\"> %s </a>" % (cmssci,cmssci))
                INDEX.write("<br /><br />\n")
            #INDEX.write("\n")


        elif dirbreg.search(NewFileLine):
            #Create a subdirectory DirectoryBrowsing to circumvent the fact the dir is not browsable if there is an index.html in it.
            os.system("mkdir %s/DirectoryBrowsing" % WebArea)
            INDEX.write("Click <a href=\"./DirectoryBrowsing/\">here</a> to browse the directory containing all results (except the root files)\n")

        elif pubdreg.search(NewFileLine):
            INDEX.write(date + "\n")
        elif candhreg.search(NewFileLine):
            for CurrentCandle in Candle:
                INDEX.write("<table cellpadding=\"20px\" border=\"1\"><tr><td>\n")
                INDEX.write("<h2>")
                INDEX.write(CurrentCandle)
                INDEX.write("</h2>\n")
                INDEX.write("<div style=\"font-size: 13\"> \n")
                for CurDir in DirName:

                    LocalPath="%s_%s" % (CurrentCandle,CurDir)
                    CandleLogFiles = getcmd("sh -c 'find %s -name \"*.log\" 2> /dev/null'" % LocalPath)
                    CandleLogFiles = filter("".__ne__,CandleLogFiles.strip().split("\n"))

                    if (len(CandleLogFiles)>0):
                        INDEX.write("<p><strong>Logfiles for %s</strong></p>\n" % CurDir)
                        for cand in CandleLogFiles:
                            print "Found %s in %s\n" % (cand,LocalPath)
                            syscp(cand,WebArea + "/.","-pR")
                            INDEX.write("<a href=\"%s\">%s </a>" % (cand,cand))
                            INDEX.write("<br />\n")

                    PrintedOnce = 0
                    for CurrentProfile in Profile:
                        for step in Step :
                            ProfileReportLink = getProfileReportLink(CurrentCandle,
                                                                     CurDir,
                                                                     step,
                                                                     CurrentProfile,
                                                                     OutputHtml[CurrentProfile])

                            if (CurrentProfile in ProfileReportLink):
                                #It could also not be there

                                if (PrintedOnce==0): 
                                    #Making sure it's printed only once per directory (TimeSize, IgProf, Valgrind) each can have multiple profiles
                                    
                                    #This is the "title" of a series of profiles, (TimeSize, IgProf, Valgrind)
                                    INDEX.write("<p><strong>%s</strong></p>\n" % CurDir)
                                    INDEX.write("<ul>\n")
                                    PrintedOnce=1
                                #Special cases first (IgProf MemAnalyse and Valgrind MemCheck)
                                if (CurrentProfile == Profile[7]):
                                    for i in range(0,3,1):
                                        ProfileReportLink = getProfileReportLink(CurrentCandle,
                                                                                 CurDir,
                                                                                 step,
                                                                                 CurrentProfile,
                                                                                 IgProfMemAnalyseOut[i])
                                        if (CurrentProfile in ProfileReportLink ) :#It could also not be there
                                            writeReportLink(INDEX,ProfileReportLink,CurrentProfile,step,NumOfEvents[CurDir],Profiler=IgProfMemAnalyseOut[i])


                                elif (CurrentProfile == Profile[9]):

                                    for i in range(0,3,1):
                                        ProfileReportLink = getProfileReportLink(CurrentCandle,
                                                                                 CurDir,
                                                                                 step,
                                                                                 CurrentProfile,
                                                                                 memcheck_valgrindOut[i])
                                        if (CurrentProfile in ProfileReportLink) : #It could also not be there
                                            writeReportLink(INDEX,ProfileReportLink,CurrentProfile,step,NumOfEvents[CurDir],Profiler=memcheck_valgrindOut[i])

                                else:
                                    writeReportLink(INDEX,ProfileReportLink,CurrentProfile,step,NumOfEvents[CurDir])


                    if PrintedOnce:
                        INDEX.write("</ul>\n")
                    PrintedOnce=0

                INDEX.write("</div>\n")
                INDEX.write("<hr />")
                INDEX.write("<br />\n")
                INDEX.write("</td></tr></table>\n")
            
        else:
            INDEX.write(NewFileLine)

    #End of while loop on template html file
    INDEX.close()

def cleanWebDir(WebArea):
    Dir = os.listdir(".")
    for CurrentDir in Dir:
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


def main():


    (options,args) = optionparse()

    repdir = os.path.normpath(".") + "/" # the report dir, may want to change this or add option in future

    #Get the CMSSW_VERSION from the environment

    (CMSSW_VERSION,
     CMSSW_RELEASE_BASE,
     CMSSW_BASE,
     HOST,
     USER,
     BASE_PERFORMANCE,
     LocalPath,
     ShowTagsResult) = get_environ()

    (WebArea,isTemp) = getWebArea(CMSSW_VERSION,UserWebArea)

    #Dump some info in a file   

    if not isEmptyWebArea(WebArea,isTemp):
        sys.exit()

    (ExecutionDate,LogFiles,date,cmsScimarkResults) = scanReportArea(repdir)

    copyReportsToWeb(WebArea)

    #Produce a small logfile with basic info on the Production area
    createLogFile("%s/ProductionLog.txt" % WebArea,HOST,date,LocalPath,ShowTagsResult)

    try:
        INDEX = createHTMLidx(WebArea,
                              BASE_PERFORMANCE,
                              CMSSW_VERSION,
                              HOST,
                              LocalPath,
                              ExecutionDate,
                              LogFiles,
                              cmsScimarkResults,
                              date)

    except IOError:
        print "Could not create index Html file for some reason"

    cleanWebDir(WebArea)

    #Creating symbolic links to the web area in subdirectory to allow directory browsing:

    createSymLink(WebArea)


    #if isTemp:
    #    os.system("rm -Rf " + os.path.dirname(WebArea))

if __name__ == "__main__":
    main()
