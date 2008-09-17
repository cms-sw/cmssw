#!/usr/bin/env python
import os, time, sys, re, glob
import optparse as opt
import cmsRelRegress as crr
from cmsPerfCommons import Candles, MIN_REQ_TS_EVENTS, CandFname, getVerFromLog

global ERRORS 
ERRORS = 0

try:
    #Get some environment variables to use
    cmssw_base        = os.environ["CMSSW_BASE"]
    cmssw_release_base= os.environ["CMSSW_RELEASE_BASE"]
    cmssw_version     = os.environ["CMSSW_VERSION"]
    host              = os.environ["HOST"]
    user              = os.environ["USER"]
except KeyError:
    print 'Error: An environment variable either CMSSW_{BASE, RELEASE_BASE or VERSION} HOST or USER is not available.'
    print '       Please run eval `scramv1 runtime -csh` to set your environment variables'
    sys.exit()

#Scripts used by the suite:
Scripts         =["cmsDriver.py","cmsRelvalreport.py","cmsRelvalreportInput.py","cmsScimark2"]
AuxiliaryScripts=["cmsScimarkLaunch.csh","cmsScimarkParser.py","cmsScimarkStop.pl"]



#Options handling
def optionParse():
    parser = opt.OptionParser(usage='''./cmsPerfSuite.py [options]
       
Examples:
./cmsPerfSuite.py
(this will run with the default options)
OR
./cmsPerfSuite.py -o "/castor/cern.ch/user/y/yourusername/yourdirectory/"
(this will archive the results in a tarball on /castor/cern.ch/user/y/yourusername/yourdirectory/)
OR
./cmsPerfSuite.py -t 5 -i 2 -v 1
(this will run the suite with 5 events for TimeSize tests, 2 for IgProf tests, 1 for Valgrind tests)
OR
./cmsPerfSuite.py -t 200 --candle QCD_80_120 --cmsdriver="--conditions FakeConditions"
(this will run the performance tests only on candle QCD_80_120, running 200 TimeSize evts, default IgProf and Valgrind evts. It will also add the option "--conditions FakeConditions" to all cmsDriver.py commands executed by the suite)
OR
./cmsPerfSuite.py -t 200 --candle QCD_80_120 --cmsdriver="--conditions=FakeConditions --eventcontent=FEVTDEBUGHLT" --step=GEN-SIM,DIGI
(this will run the performance tests only on candle QCD_80_120, running 200 TimeSize evts, default IgProf and Valgrind evts. It will also add the option "--conditions=FakeConditions" and the option "--eventcontent=FEVTDEBUGHLT" to all cmsDriver.py commands executed by the suite. In addition it will run only 2 cmsDriver.py "steps": "GEN,SIM" and "DIGI". Note the syntax GEN-SIM for combined cmsDriver.py steps)

Legal entries for individual candles (--candle option):
%s
''' % ("\n".join(Candles)))

    parser.set_defaults(TimeSizeEvents   = 100,
                        IgProfEvents     = 5,
                        ValgrindEvents   = 1,
                        cmsScimark       = 10,
                        cmsScimarkLarge  = 10,
                        cmsdriverOptions = "",
                        stepOptions      = "",
                        candleOptions    = "",
                        quicktest        = False,
                        unittest         = False,
                        verbose          = True,
                        castordir = "/castor/cern.ch/cms/store/relval/performance/",
                        cores     = 4, #Number of cpu cores on the machine
                        cpu       = 1) #Cpu core on which the suite is run:

    devel  = opt.OptionGroup(parser, "Developer Options",
                                     "Caution: use these options at your own risk."
                                     "It is believed that some of them bite.\n")

    devel.add_option(
        '-d',
        '--debug',
        action='store_true',
        dest='debug',
        help='Debug',
        #metavar='<DIR>',
        )

    devel.add_option(
        '--quicktest',
        action="store_true",
        dest='quicktest',
        help='Quick overwrite all the defaults to small numbers so that we can run a quick test of our chosing.',
        #metavar='<#EVENTS>',
        )  

    devel.add_option(
        '--test',
        action="store_true",
        dest='unittest',
        help='Perform a simple test, overrides other options. Overrides verbosity and sets it to false.',
        #metavar='<#EVENTS>',
        )    

    parser.add_option(
        '-o',
        '--output',
        type='string',
        dest='castordir',
        help='specify the wanted CASTOR directory where to store the results tarball',
        metavar='<DIR>',
        )

    parser.add_option(
        '-r',
        '--prevrel',
        type='string',
        dest='previousrel',
        default="",
        help='Top level dir of previous release for regression analysis',
        metavar='<DIR>',
        )
        
    parser.add_option(
        '-t',
        '--timesize',
        type='int',
        dest='TimeSizeEvents',
        help='specify the number of events for the TimeSize tests',
        metavar='<#EVENTS>',
        )

    parser.add_option(
        '-q',
        '--quiet',
        action="store_false",
        dest='verbose',
        help='Output less information',
        #metavar='<#EVENTS>',
        )
    
    parser.add_option(
        '-i',
        '--igprof',
        type='int',
        dest='IgProfEvents',
        help='specify the number of events for the IgProf tests',
        metavar='<#EVENTS>',
        )

    parser.add_option(
        '-v',
        '--valgrind',
        type='int',
        dest='ValgrindEvents',
        help='specify the number of events for the Valgrind tests',
        metavar='<#EVENTS>',
        )

    parser.add_option(
        '--cmsScimark',
        type='int',
        dest='cmsScimark',
        help='specify the number of times the cmsScimark benchmark is run before and after the performance suite on cpu1',
        metavar='',
        )

    parser.add_option(
        '--cmsScimarkLarge',
        type='int',
        dest='cmsScimarkLarge',
        help='specify the number of times the cmsScimarkLarge benchmark is run before and after the performance suite on cpu1',
        metavar='',
        )

    parser.add_option(
        '-c',
        '--cmsdriver',
        type='string',
        dest='cmsdriverOptions',
        help='specify special options to use with the cmsDriver.py commands (designed for integration build use',
        metavar='<OPTION_STR>',
        )

    parser.add_option(
        '--step',
        type='string',
        dest='stepOptions',
        help='specify the processing steps intended (instead of the default ones)',
        metavar='<OPTION_STR>',
        )

    parser.add_option(
        '--candle',
        type='string',
        dest='candleOptions',
        help='specify the candle(s) to run (instead of all 7 default candles)',
        metavar='<OPTION_STR>',
        )

    parser.add_option(
        '--cpu',
        type='string',
        dest='cpu',
        help='specify the core on which to run the performance suite',
        metavar='<CPU_STR>',
        )

    parser.add_option(
        '--cores',
        type='string',
        dest='cores',
        help='specify the number of cores of the machine (can be used with 0 to stop cmsScimark from running on the other cores)',
        metavar='<OPTION_STR>',
        )

    parser.add_option_group(devel)        

    return parser.parse_args()

def usage():
    return __doc__

def runCmdSet(cmd):
    exitstat = None
    if len(cmd) <= 1:
        exitstat = runcmd(cmd)
        if _verbose:
            printFlush(cmd)
    else:
        for subcmd in cmd:
            if _verbose:
                printFlush(subcmd)
        exitstat = runcmd(" && ".join(cmd))
    if _verbose:
        printFlush(getDate())
    return exitstat

def printFlush(command):
    if _verbose:
        print command
    sys.stdout.flush()

def runcmd(command):
    process  = os.popen(command)
    cmdout   = process.read()
    exitstat = process.close()
    if _verbose:
        print cmdout
    return exitstat

def getDate():
    return time.ctime()

def benchmarks(name,bencher):
    cmd = Commands[3]
    redirect = ""
    if name == "cmsScimark2.log":
        redirect = " >& "
    else:
        redirect = " -large >& "
        
    numofbenchs = int(bencher)
    for i in range(numofbenchs):
        command= cmd + redirect + name
        printFlush(command+" [%s/%s]"%(i+1,numofbenchs))
        runcmd(command)
        sys.stdout.flush()


def printDate():
    print getDate()

def getPrereqRoot(rootdir,rootfile):
    print "WARNING: %s file required to run QCD profiling does not exist. Now running cmsDriver.py to get Required Minbias root file"   % (rootdir + "/" +rootfile)

    if not os.path.exists(rootdir):
        os.system("mkdir -p %s" % rootdir)
    if not _debug:
        cmd = "cd %s ; cmsDriver.py MinBias_cfi -s GEN,SIM -n %s >& ../minbias_for_pileup_generate.log" % (rootdir,str(10))
        print cmd
        os.system(cmd)
    if not os.path.exists(rootdir + "/" + rootfile):
        print "ERROR: We can not run QCD profiling please create root file %s to run QCD profiling." % (rootdir + "/" + rootfile)


def checkQcdConditions(candles,TimeSizeEvents,rootdir,rootfile):
    if TimeSizeEvents < MIN_REQ_TS_EVENTS :
        print "WARNING: TimeSizeEvents is less than %s but QCD needs at least that to run. PILE-UP will be ignored" % MIN_REQ_TS_EVENTS
        
        
    rootfilepath = rootdir + "/" + rootfile
    if not os.path.exists(rootfilepath):
        getPrereqRoot(rootdir,rootfile)
        if not os.path.exists(rootfilepath) and not _debug:
            print "ERROR: Could not create or find a rootfile %s with enough TimeSize events for QCD exiting..." % rootfilepath
            sys.exit()
    else:
        print "%s Root file for QCD exists. Good!!!" % (rootdir + "/" + rootfile)
    return candles

def mkCandleDir(candle,profiler):
    dir = "%s_%s" % (candle,profiler)
    runcmd( "mkdir -p %s" % dir )
    if _verbose:
        printDate()
    #runCmdSet(cmd)
    return dir

def cpIgProfGenSim(dir,candle):
    cmds = ("cd %s" % dir,
            "cp -pR ../%s_IgProf/%s_GEN,SIM.root ."  % (candle,candle))
    runCmdSet(cmds)

def displayErrors(file):
    global ERRORS
    try:
        for line in open(file,"r"):
            if "cerr" in line:
                print "ERROR: %s" % line
                ERRORS += 1
    except OSError, detail:
        print "WARNING: %s" % detail
        ERRORS += 1        
    except IOError, detail:
        print "WARNING: %s" % detail
        ERRORS += 1
    

def valFilterReport(dir,cmsver):
    cmds = ("cd %s" % dir,
            "grep -v \"step=GEN,SIM\" SimulationCandles_%s.txt > tmp" % (cmssw_version),
            "mv tmp SimulationCandles_%s.txt"                         % (cmssw_version))
    runCmdSet(cmds)

def runCmsReport(dir,cmsver,candle):
    cmd  = Commands[1]
    cmds = ("cd %s"                 % (dir),
            "%s -i SimulationCandles_%s.txt -t perfreport_tmp -R -P >& %s.log" % (cmd,cmsver,candle))
    exitstat = None
    if not _debug:
        exitstat = runCmdSet(cmds)
        
    if _unittest and (not exitstat == None):
        print "ERROR: CMS Report returned a non-zero exit status "
        sys.exit()

def testCmsDriver(dir,cmsver,candle):
    cmd  = Commands[0]
    noExit = True
    stepreg = re.compile("--step=([^ ]*)")
    for line in open("./%s/SimulationCandles_%s.txt" % (dir,cmsver)):
        if (not line.lstrip().startswith("#")) and not (line.isspace() or len(line) == 0): 
            cmdonline  = line.split("@@@",1)[0]
            stepbeingrun = "Unknown"
            matches = stepreg.search(cmdonline)
            if not matches == None:
                stepbeingrun = matches.groups()[0]
            if "PILEUP" in cmdonline:
                stepbeingrun += "_PILEUP"
            print cmdonline
            cmds = ("cd %s"      % (dir),
                    "%s  >& ../cmsdriver_unit_test_%s_%s.log"    % (cmdonline,candle,stepbeingrun))
            out = runCmdSet(cmds) 
            if not out == None:
                sig     = out >> 16    # Get the top 16 bits
                xstatus = out & 0xffff # Mask out all bits except the first 16 
                print "FATAL ERROR: CMS Driver returned a non-zero exit status (which is %s) when running %s for candle %s. Signal interrupt was %s" % (xstatus,stepbeingrun,candle,sig)
                sys.exit()
    

def runCmsInput(dir,numevents,candle,cmsdrvopts,stepopt,profiler):
    cmd = Commands[2]
    profilers = { "TimeSize" : "0123",
                  "IgProf"   : "4567",
                  "Valgrind" : "89"  ,
                  "None"     : "-1"  } 

    cmds = ("cd %s"                 % (dir),
            "%s %s \"%s\" %s %s %s" % (cmd,
                                       numevents,
                                       candle,
                                       profilers[profiler],
                                       cmsdrvopts,
                                       stepopt))
    exitstat = runCmdSet(cmds)
    if _unittest and (not exitstat == None):
        print "ERROR: CMS Report Input returned a non-zero exit status " 

def simpleGenReport(NumEvents,candles,cmsdriverOptions,stepOptions,cmssw_version,Name):
    valgrind = Name == "Valgrind"

        
    for candle in candles:
        adir = mkCandleDir(candle,Name)
        if valgrind:
            if candle == "SingleMuMinusPt10" : 
                print "Valgrind tests **GEN,SIM ONLY** on %s candle" % candle                
            else:
                print "Valgrind tests **SKIPPING GEN,SIM** on %s candle" % candle                
                cpIgProfGenSim(adir,candle)                

        if _unittest:
            # Run cmsDriver.py
            runCmsInput(adir,NumEvents,candle,cmsdriverOptions,stepOptions,"None")
            testCmsDriver(adir,cmssw_version,candle)
        else:
            runCmsInput(adir,NumEvents,candle,cmsdriverOptions,stepOptions,Name)            
            if valgrind:
                valFilterReport(adir,cmssw_version)             
            runCmsReport(adir,cmssw_version,candle)
            proflogs = []
            if   Name == "TimeSize":
                proflogs = [ "TimingReport" ]
            elif Name == "Valgrind":
                pass
            elif Name == "IgProf":
                pass
                
            for proflog in proflogs:
                globpath = os.path.join(adir,"%s_*_%s.log" % (CandFname[candle],proflog))
                print "Looking for logs that match", globpath
                logs     = glob.glob(globpath)
                for log in logs:
                    print "Found log", log
                    displayErrors(log)

def main(argv):
    #Some default values:

    #Let's check the command line arguments

    (options, args) = optionParse()

    global _debug, _unittest, _verbose
    _debug           = options.debug
    castordir        = options.castordir
    TimeSizeEvents   = options.TimeSizeEvents
    IgProfEvents     = options.IgProfEvents
    ValgrindEvents   = options.ValgrindEvents
    cmsScimark       = options.cmsScimark
    cmsScimarkLarge  = options.cmsScimarkLarge
    cmsdriverOptions = options.cmsdriverOptions
    stepOptions      = options.stepOptions
    quicktest        = options.quicktest
    candleoption     = options.candleOptions.upper()
    cpu              = options.cpu
    cores            = options.cores
    _unittest        = options.unittest 
    _verbose         = options.verbose
    prevrel          = options.previousrel

    if not prevrel == "":
        prevrel = os.path.abspath(prevrel)
        if not os.path.exists(prevrel):
            print "ERROR: Previous release dir %s could not be found" % prevrel
            sys.exit()

    if quicktest:
        TimeSizeEvents = 1
        IgProfEvents = 1
        ValgrindEvents = 0
        cmsScimark = 1
        cmsScimarkLarge = 1

    if _unittest:
        _verbose = False
        if candleoption == "":
            candleoption = "MinBias"
        if stepOptions == "":
            stepOptions = "GEN-SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI-RECO"
        cmsScimark      = 0
        cmsScimarkLarge = 0
        ValgrindEvents  = 0
        IgProfEvents    = 0
        TimeSizeEvents  = 1
    if not cmsdriverOptions == "":
        cmsdriverOptions = "--cmsdriver=" + cmsdriverOptions
    #Case with no arguments (using defaults)
    if options == []:
        print "No arguments given, so DEFAULT test will be run:"
        
    #Print a time stamp at the beginning:

    path=os.path.abspath(".")
    print "Performance Suite started running at %s on %s in directory %s, run by user %s" % (getDate(),host,path,user)
    showtags=os.popen4("showtags -r")[1].read()
    print showtags
    
    #For the log:
    if _verbose:
        print "The performance suite results tarball will be stored in CASTOR at %s" % castordir
        print "%s TimeSize events" % TimeSizeEvents
        print "%s IgProf events"   % IgProfEvents
        print "%s Valgrind events" % ValgrindEvents
        print "%s cmsScimark benchmarks before starting the tests"      % cmsScimark
        print "%s cmsScimarkLarge benchmarks before starting the tests" % cmsScimarkLarge
        
    if cmsdriverOptions != "":
        print "Running cmsDriver.py with the special user defined options: %s" % cmsdriverOptions
        
        #Wrapping the options with "" for the cmsSimPyRelVal.pl until .py developed
        cmsdriverOptions= '"%s"' % (cmsdriverOptions)
    if stepOptions !="":
        print "Running user defined steps only: %s" % stepOptions
        
        #Wrapping the options with "" for the cmsSimPyRelVal.pl until .py developed
        stepOptions='"--usersteps=%s"' % (stepOptions)
    if candleoption !="":
        print "Running only %s candle, instead of the whole suite" % candleoption
    print "This machine ( %s ) is assumed to have %s cores, and the suite will be run on cpu %s" %(host,cores,cpu)
    
    #Actual script actions!
    #Will have to fix the issue with the matplotlib pie-charts:
    #Used to source /afs/cern.ch/user/d/dpiparo/w0/perfreport2.1installation/share/perfreport/init_matplotlib.sh
    #Need an alternative in the release

    #Command Handling:
    global Commands
    
    Commands=[]
    AuxiliaryCommands=[]
    AllScripts=Scripts+AuxiliaryScripts

    for script in AllScripts:
        which="which "+script
        
        #Logging the actual version of cmsDriver.py, cmsRelvalreport.py, cmsSimPyRelVal.pl
        whichstdout=os.popen4(which)[1].read()
        print whichstdout
        if script in Scripts:
            command="taskset -c "+str(cpu)+" "+script
            Commands.append(command)
        elif script == "cmsScimarkLaunch.csh":
            for core in range(int(cores)):
                if core != int(cpu):
                    command="taskset -c %s %s %s" % (str(core),script,str(core))
                    AuxiliaryCommands.append(command)
        else:
            command=script
            AuxiliaryCommands.append(command)
            
    sys.stdout.flush()
    
    #First submit the cmsScimark benchmarks on the unused cores:
    scimark = ""
    scimarklarge = ""
    if not _unittest:    
        for core in range(int(cores)):
            if core != int(cpu):
                print "Submitting cmsScimarkLaunch.csh to run on core cpu"+str(core)
                command="taskset -c %s cmsScimarkLaunch.csh %s &" % (str(core),str(core))
                print command
            
                #cmsScimarkLaunch.csh is an infinite loop to spawn cmsScimark2 on the other
                #cpus so it makes no sense to try reading its stdout/err 
                os.popen4(command)
            
    #dont do benchmarking if in debug mode... saves time
    benching = not _debug
    if benching and not _unittest:
        #Submit the cmsScimark benchmarks on the cpu where the suite will be run:        
        scimark      = open("cmsScimark2.log"      ,"w")        
        scimarklarge = open("cmsScimark2_large.log","w")
        print "Starting with %s cmsScimark on cpu%s"%(cmsScimark,cpu)
        benchmarks(scimark.name,cmsScimark)
    
        print "Following with %s cmsScimarkLarge on cpu%s"%(cmsScimarkLarge,cpu)
        benchmarks(scimarklarge.name,cmsScimarkLarge)
        
    #Here the real performance suite starts
    #List of Candles


    AllCandles=Candles

    isAllCandles = candleoption == ""
    candles = {}
    if isAllCandles:
        candles=AllCandles
    else:
        candles=candleoption.split(",")

    qcdWillRun = (not isAllCandles) and "QCD_80_120" in candles 
    if qcdWillRun:
        candles = checkQcdConditions(candles,
                                     TimeSizeEvents,
                                     "./%s_%s" % ("MinBias","TimeSize"),
                                     "%s_cfi_GEN_SIM.root" % "MinBias")

    #TimeSize tests:
    if TimeSizeEvents > 0:
        print "Launching the TimeSize tests (TimingReport, TimeReport, SimpleMemoryCheck, EdmSize) with %s events each" % TimeSizeEvents
        printDate()
        sys.stdout.flush()
        simpleGenReport(TimeSizeEvents,candles,cmsdriverOptions,stepOptions,cmssw_version,"TimeSize")

    #IgProf tests:
    if IgProfEvents > 0:
        print "Launching the IgProf tests (IgProfPerf, IgProfMemTotal, IgProfMemLive, IgProfMemAnalyse) with %s events each" % IgProfEvents
        printDate()
        IgCandles = candles
        sys.stdout.flush()
        #By default run IgProf only on QCD_80_120 candle
        if isAllCandles:
            IgCandles = [ "QCD_80_120" ]
        simpleGenReport(IgProfEvents,IgCandles,cmsdriverOptions,stepOptions,cmssw_version,"IgProf")

    #Valgrind tests:
    if ValgrindEvents > 0:
        print "Launching the Valgrind tests (callgrind_FCE, memcheck) with %s events each" % ValgrindEvents
        printDate()   
        valCandles = candles
        
        if isAllCandles:
            cmds=[]
            sys.stdout.flush()    
            #By default run Valgrind only on QCD_80_120, skipping SIM step since it would take forever (and do SIM step on SingleMu)
            valCandles = [ "QCD_80_120" ]

        #Besides always run, only once the GEN,SIM step on SingleMu:
        valCandles.append("SingleMuMinusPt10")
        #In the user-defined candles a different behavior: do Valgrind for all specified candles (usually it will only be 1)
        #usercandles=candleoption.split(",")
        simpleGenReport(ValgrindEvents,valCandles,cmsdriverOptions,stepOptions,cmssw_version,"Valgrind")

    if benching and not _unittest:
    #Ending the performance suite with the cmsScimark benchmarks again: 
        print "Ending with %s cmsScimark on cpu%s"%(cmsScimark,cpu)
        benchmarks(scimark.name,cmsScimark)
    
        print "Following with %s cmsScimarkLarge on cpu%s"%(cmsScimarkLarge,cpu)
        benchmarks(scimarklarge.name,cmsScimarkLarge)
    
    #Stopping all cmsScimark jobs and analysing automatically the logfiles
    print "Stopping all cmsScimark jobs"
    printFlush(AuxiliaryScripts[2])
    printFlush(os.popen4(AuxiliaryScripts[2])[1].read())

    if not prevrel == "":
        crr.regressReports(prevrel,os.path.abspath("./"),oldRelName = getVerFromLog(prevrel),newRelName=cmssw_version)

    #Create a tarball of the work directory
    TarFile = cmssw_version + "_"     +     host    + "_"     + user + ".tar"
    tarcmd  = "tar -cvf "   + TarFile + " *; gzip " + TarFile
    printFlush(tarcmd)
    printFlush(os.popen4(tarcmd)[1].read())
    
    #Archive it on CASTOR
    castorcmd="rfcp %s.gz %s.gz" % (TarFile,os.path.join(castordir,TarFile))
    
    printFlush(castorcmd)
    printFlush(os.popen4(castorcmd)[1].read())

    #End of script actions!

    #Print a time stamp at the end:
    date=time.ctime(time.time())
    print "Performance Suite finished running at %s on %s in directory %s" % (date,host,path)
    if ERRORS == 0:
        print "There were no errors detected in any of the log files!"
    else:
        print "ERROR: There were %s errors detected in the log files, please revise!" % ERRORS


if __name__ == "__main__":
    main(sys.argv[1:])

