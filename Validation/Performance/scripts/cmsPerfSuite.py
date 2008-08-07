#!/usr/bin/python
'''
Usage: ./cmsPerfSuite.py [options]
       
Options:

Examples:
./cmsPerfSuite.py
(this will run with the default options)
OR
./cmsPerfSuite.py -o "/castor/cern.ch/user/y/yourusername/yourdirectory/"
(this will archive the results in a tarball on /castor/cern.ch/user/y/yourusername/yourdirectory/)
OR
./cmsPerfSuite.py -t 5 -i 2 -v 1
(this will run the suite with 5 events for TimeSize tests, 2 for IgProf tests, 0 for Valgrind tests)
OR
./cmsPerfSuite.py -t 200 --candle QCD_80_120 --cmsdriver="--conditions FakeConditions"
(this will run the performance tests only on candle QCD_80_120, running 200 TimeSize evts, default IgProf and Valgrind evts. It will also add the option "--conditions FakeConditions" to all cmsDriver.py commands executed by the suite)
OR
./cmsPerfSuite.py -t 200 --candle QCD_80_120 --cmsdriver="--conditions=FakeConditions --eventcontent=FEVTDEBUGHLT" --step=GEN-SIM,DIGI
(this will run the performance tests only on candle QCD_80_120, running 200 TimeSize evts, default IgProf and Valgrind evts. It will also add the option "--conditions=FakeConditions" and the option "--eventcontent=FEVTDEBUGHLT" to all cmsDriver.py commands executed by the suite. In addition it will run only 2 cmsDriver.py "steps": "GEN,SIM" and "DIGI". Note the syntax GEN-SIM for combined cmsDriver.py steps)

Legal entries for individual candles (--candle option):
HiggsZZ4LM200
MinBias
SingleElectronE1000
SingleMuMinusPt10
SinglePiMinusE1000
TTbar
QCD_80_120
'''
import os
import time
import getopt
import sys
import optparse as opt

global ERRORS 
ERRORS = 0

def optionParse():
    parser = opt.OptionParser(usage=usage())

    parser.set_defaults(TimeSizeEvents   = 100,
                        IgProfEvents     = 5,
                        ValgrindEvents   = 1,
                        cmsScimark       = 10,
                        cmsScimarkLarge  = 10,
                        cmsdriverOptions = "",
                        stepOptions      = "",
                        candleOptions    = "",
                        castordir = "/castor/cern.ch/cms/store/relval/performance/",
                        cores=4, #Number of cpu cores on the machine
                        cpu=1) #Cpu core on which the suite is run:

    parser.add_option(
        '-d',
        '--debug',
        action='store_true',
        dest='debug',
        help='Debug',
        #metavar='<DIR>',
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
        '-t',
        '--timesize',
        type='int',
        dest='TimeSizeEvents',
        help='specify the number of events for the TimeSize tests',
        metavar='<#EVENTS>',
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
        type='string',
        dest='cmsScimark',
        help='specify the number of times the cmsScimark benchmark is run before and after the performance suite on cpu1',
        metavar='',
        )

    parser.add_option(
        '--cmsScimarkLarge',
        type='string',
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
    return parser.parse_args()

MIN_REQ_TS_EVENTS = 8

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
#Some defaults:


#Options handling


def usage():
    print __doc__

def runCmdSet(cmd):
    if len(cmd) <= 1:
        runcmd(cmd)
        printFlush(cmd)
    else:
        for subcmd in cmd:
            printFlush(subcmd)
        runcmd(";".join(cmd))
    printFlush(getDate())

def printFlush(command):
    print command
    sys.stdout.flush()

def runcmd(command):
    cmdout=os.popen4(command)[1].read()
    print cmdout

def benchmarks(cmd,redirect,name,bencher):
    numofbenchs = int(bencher)
    for i in range(numofbenchs):
        command= cmd + redirect + name
        printFlush(command+" [%s/%s]"%(i+1,numofbenchs))
        runcmd(command)
        sys.stdout.flush()

def getDate():
    return time.ctime()

def printDate():
    print getDate()

def getPrereqRoot(rootdir,rootfile):
    print "ERROR: %s file required to run QCD profiling does not exist. We can not run QCD profiling please create root file" % (rootdir + "/" + rootfile)
    print "       to run QCD profiling."
    print "       Running cmsDriver.py to get Required MinbiasEvents"
    if not os.path.exists(rootdir):
        os.system("mkdir -p %s" % rootdir)
    if not _debug:
        os.system("cd %s ; cmsDriver.py MinBias_cfi -s GEN,SIM -n 10" % (rootdir))


def checkQcdConditions(isAllCandles,candles,TimeSizeEvents,rootdir,rootfile):
    if TimeSizeEvents < MIN_REQ_TS_EVENTS :
        print "WARNING: TimeSizeEvents is less than 8 but QCD needs at least that to run. Setting TimeSizeEvents to 8"
        if isAllCandles:
            candles.remove("QCD_80_120")
        else:
            candles.remove("QCD_80_120")
        
    rootfilepath = rootdir + "/" + rootfile
    if not os.path.exists(rootfilepath):
        getPrereqRoot(rootdir,rootfile)
        if not os.path.exists(rootfilepath) and not _debug:
            print "ERROR: Could not create or find a rootfile %s with enough TimeSize events for QCD exiting..." % rootfilepath
            sys.exit()
    return candles

def mkCandleDir(candle,profiler):
    dir = "%s_%s" % (candle,profiler)
    runcmd( "mkdir -p %s" % dir )
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
    except IOError:
        print "WARNING: The log file could not be open for some reason"
        ERRORS += 1
    

def valFilterReport(dir,cmsver):
    cmds = ("cd %s" % dir,
            "grep -v \"step=GEN,SIM\" SimulationCandles_%s.txt > tmp" % (cmssw_version),
            "mv tmp SimulationCandles_%s.txt"                         % (cmssw_version))
    runCmdSet(cmds)

def runCmsReport(dir,cmd,cmsver,candle):
    cmds = ("cd %s"                 % (dir),
            "%s -i SimulationCandles_%s.txt -t perfreport_tmp -R -P >& %s.log" % (cmd,cmsver,candle))
    if not _debug:
        runCmdSet(cmds)

def runCmsInput(dir,cmd,numevents,candle,cmsdrvopts,stepopt,profiler):

    profilers = { "TimeSize" : "0123",
                  "IgProf"   : "4567",
                  "Valgrind" : "89"} 

    cmds = ("cd %s"                 % (dir),
            "%s %s \"%s\" %s %s %s" % (cmd,
                                       numevents,
                                       candle,
                                       profilers[profiler],
                                       cmsdrvopts,
                                       stepopt))
    runCmdSet(cmds)

def main(argv):
    #Some default values:

    #Let's check the command line arguments

    (options, args) = optionParse()

    global _debug  
    _debug         = options.debug
    castordir      = options.castordir
    TimeSizeEvents = options.TimeSizeEvents
    IgProfEvents   = options.IgProfEvents
    ValgrindEvents = options.ValgrindEvents
    cmsScimark     = options.cmsScimark
    cmsScimarkLarge= options.cmsScimarkLarge
    cmsdriverOptions=options.cmsdriverOptions
    stepOptions    = options.stepOptions
    candleoption   = options.candleOptions
    cpu            = options.cpu
    cores          = options.cores


        #opts, args = getopt.getopt(argv, "o:t:i:v:hd", ["output=","timesize=","igprof=","valgrind=","cmsScimark=","cmsScimarkLarge=","cmsdriver=","step=","candle=","cpu=","cores=","help"])
   # except getopt.GetoptError:
 
    #Case with no arguments (using defaults)
    if options == []:
        print "No arguments given, so DEFAULT test will be run:"
        
    #Print a time stamp at the beginning:

    path=os.path.abspath(".")
    print "Performance Suite started running at %s on %s in directory %s, run by user %s" % (getDate(),host,path,user)
    showtags=os.popen4("showtags -r")[1].read()
    print showtags
    
    #For the log:
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
                    command="taskset -c "+str(core)+" "+script+" "+str(core)
                    AuxiliaryCommands.append(command)
        else:
            command=script
            AuxiliaryCommands.append(command)
            
    #print Commands
    #print AuxiliaryCommands
    sys.stdout.flush()
    
    #First submit the cmsScimark benchmarks on the unused cores:
    for core in range(int(cores)):
        if core != int(cpu):
            print "Submitting cmsScimarkLaunch.csh to run on core cpu"+str(core)
            command="taskset -c "+str(core)+" cmsScimarkLaunch.csh "+str(core)+"&"
            print command
            
            #cmsScimarkLaunch.csh is an infinite loop to spawn cmsScimark2 on the other
            #cpus so it makes no sense to try reading its stdout/err 
            os.popen4(command)
            
    #Submit the cmsScimark benchmarks on the cpu where the suite will be run:
    scimark      = open("cmsScimark2.log"      ,"w")
    scimarklarge = open("cmsScimark2_large.log","w")


    #dont do benchmarking if in debug mode... saves time
    benching = not _debug
    if benching:
        print "Starting with %s cmsScimark on cpu%s"%(cmsScimark,cpu)
        benchmarks(Commands[3]," >& ",scimark.name,cmsScimark)
    
        print "Following with %s cmsScimarkLarge on cpu%s"%(cmsScimarkLarge,cpu)
        benchmarks(Commands[3]," -large >& ",scimarklarge.name,cmsScimarkLarge)
        
    #Here the real performance suite starts
    #List of Candles
    Candles={"HiggsZZ4LM200"      : "HZZLLLL",
             "MinBias"            : "MINBIAS",
             "SingleElectronE1000": "E -e 1000",
             "SingleMuMinusPt10"  : "MU- -e pt10",
             "SinglePiMinusE1000" : "PI- -e 1000",
             "TTbar"              : "TTBAR",
             "QCD_80_120"         : "QCD -e 80_120"
             }
    AllCandles=Candles.keys()
    
    #Sort the candles to make sure MinBias is executed before QCD_80_120, otherwise DIGI PILEUP would not find its MinBias root files
    AllCandles.sort()

    isAllCandles = candleoption == ""
    
    candles = {}
    if isAllCandles:
        candles=AllCandles
    else:
        candles=candleoption.split(",")

    qcdWillRun = isAllCandles or ((not isAllCandles) and "QCD_80_120" in usercandles )

    if qcdWillRun:
        candles = checkQcdConditions(isAllCandles,
                                     candles,
                                     TimeSizeEvents,
                                     "./%s_%s" % ("MinBias","TimeSize"),
                                     "%s_cfi_GEN_SIM.root" % "MinBias")  

    #TimeSize tests:
    if int(TimeSizeEvents)>0:

        print "Launching the TimeSize tests (TimingReport, TimeReport, SimpleMemoryCheck, EdmSize) with %s events each" % TimeSizeEvents
        printDate()
        
        sys.stdout.flush()
        for candle in candles:
            dir = mkCandleDir(candle,"TimeSize")
            runCmsInput(dir,Commands[2],TimeSizeEvents,Candles[candle],cmsdriverOptions,stepOptions,"TimeSize")
            runCmsReport(dir,Commands[1],cmssw_version,candle)
            displayErrors("%s/%s_%s.log" % (dir,candle,"TimeSize"))

    #IgProf tests:
    if int(IgProfEvents)>0:
        print "Launching the IgProf tests (IgProfPerf, IgProfMemTotal, IgProfMemLive, IgProfMemAnalyse) with %s events each" % IgProfEvents
        printDate()

        IgCandles = candles
        cmds=[]
        sys.stdout.flush()        
        if isAllCandles:
            #By default run IgProf only on QCD_80_120 candle
            IgCandles = [ "QCD_80_120" ]


        for candle in IgCandles:
            dir = mkCandleDir(candle,"IgProf")
            runCmsInput(dir,Commands[2],IgProfEvents,Candles[candle],cmsdriverOptions,stepOptions,"IgProf")
            runCmsReport(dir,Commands[1],cmssw_version,candle)
            displayErrors("%s/%s_%s.log" % (dir,candle,"IgProf"))            

    #Valgrind tests:
    if int(ValgrindEvents)>0:
        print "Launching the Valgrind tests (callgrind_FCE, memcheck) with %s events each" % ValgrindEvents
        printDate()   

        valCandles = candles
        
        if isAllCandles:
            cmds=[]
            sys.stdout.flush()
            
            #By default run Valgrind only on QCD_80_120, skipping SIM step since it would take forever (and do SIM step on SingleMu)
            valCandles = [ "QCD_80_120" ]

            
        #In the user-defined candles a different behavior: do Valgrind for all specified candles (usually it will only be 1)
        #usercandles=candleoption.split(",")
        for candle in valCandles:
            print "Valgrind tests **SKIPPING GEN,SIM** on %s candle" % candle
            dir = mkCandleDir(candle,"Valgrind")
            cpIgProfGenSim(dir,candle)
            runCmsInput(dir,Commands[2],ValgrindEvents,Candles[candle],cmsdriverOptions,stepOptions,"Valgrind")
            valFilterReport(dir,cmssw_version)
            runCmsReport(dir,Commands[1],cmssw_version,candle)
            displayErrors("%s/%s_%s.log" % (dir,candle,"Valgrind")) 

        #Besides always run, only once the GEN,SIM step on SingleMu:
        candle = "SingleMuMinusPt10"
        print "Valgrind tests **GEN,SIM ONLY** on %s candle" % candle

        dir = mkCandleDir(candle,"Valgrind")
        runCmsInput(dir,Commands[2],ValgrindEvents,Candles[candle],cmsdriverOptions,stepOptions,"Valgrind")
        valFilterReport(dir,cmssw_version)
        runCmsReport(dir,Commands[1],cmssw_version,candle)
        displayErrors("%s/%s_%s.log" % (dir,candle,"Valgrind"))            
        

    if benching:
    #Ending the performance suite with the cmsScimark benchmarks again: 
        print "Ending with %s cmsScimark on cpu%s"%(cmsScimark,cpu)
        benchmarks(" >& ",scimark.name,cmsScimark)
    
        print "Following with %s cmsScimarkLarge on cpu%s"%(cmsScimarkLarge,cpu)
        benchmarks(" -large >& ",scimarklarge.name,cmsScimarkLarge)
    
    #Stopping all cmsScimark jobs and analysing automatically the logfiles
    print "Stopping all cmsScimark jobs"
    printFlush(AuxiliaryScripts[2])

    printFlush(os.popen4(AuxiliaryScripts[2])[1].read())

    #Create a tarball of the work directory
    TarFile = cmssw_version + "_"     +     host    + "_"     + user + ".tar"
    tarcmd  = "tar -cvf "   + TarFile + " *; gzip " + TarFile
    printFlush(tarcmd)
    printFlush(os.popen4(tarcmd)[1].read())
    
    #Archive it on CASTOR
    castorcmd="rfcp " + TarFile + ".gz " + castordir + TarFile+".gz"
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

