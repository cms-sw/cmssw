#!/usr/bin/env python
import os, time, sys, re, glob, exceptions
import optparse as opt
import cmsRelRegress as crr
from cmsPerfCommons import Candles, MIN_REQ_TS_EVENTS, KeywordToCfi, CandFname, getVerFromLog
import cmsRelValCmd,cmsCpuInfo
import threading #Needed in threading use for Valgrind
import subprocess #Nicer subprocess management than os.popen

class PerfThread(threading.Thread):
    def __init__(self,**args):
        self.args=args
        threading.Thread.__init__(self)
    def run(self):
        self.suite=PerfSuite()
        #print "Arguments inside the thread instance:"
        #print type(self.args)
        #print self.args
        self.suite.runPerfSuite(**(self.args))#self.args)

class ValgrindThread(threading.Thread):
    def __init__(self,valgrindArgs): #valgrindArgs should be selecting CallGrind/MemCheck, Candle, NumOfEvent
        self.valgrindArgs=valgrindArgs
        threading.Thread.__init__(self)
    def run(self):
        print self
        
class PerfSuite:
    def __init__(self):
        
        self.ERRORS = 0
        self._CASTOR_DIR = "/castor/cern.ch/cms/store/relval/performance/"
        self._dryrun   = False
        self._debug    = False
        self._unittest = False
        self._verbose  = True
        self.logh = sys.stdout
    
        #Get some environment variables to use
        try:
            self.cmssw_version= os.environ["CMSSW_VERSION"]
            self.host         = os.environ["HOST"]
            self.user              = os.environ["USER"]
        except KeyError:
            print 'Error: An environment variable either CMSSW_{BASE, RELEASE_BASE or VERSION} HOST or USER is not available.'
            print '       Please run eval `scramv1 runtime -csh` to set your environment variables'
            sys.exit()
    
        #Scripts used by the suite:
        self.Scripts         =["cmsDriver.py","cmsRelvalreport.py","cmsRelvalreportInput.py","cmsScimark2"]
        self.AuxiliaryScripts=["cmsScimarkLaunch.csh","cmsScimarkParser.py","cmsScimarkStop.py"]
    
    
    #Options handling
    def optionParse(self,argslist=None):
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
    
        parser.set_defaults(TimeSizeEvents   = 100        ,
                            IgProfEvents     = 5          ,
                            ValgrindEvents   = 1          ,
                            cmsScimark       = 10         ,
                            cmsScimarkLarge  = 10         ,  
                            cmsdriverOptions = "--eventcontent FEVTDEBUGHLT", # Decided to avoid using the automatic parsing of cmsDriver_highstats_hlt.txt: cmsRelValCmd.get_cmsDriverOptions(), #Get these options automatically now!
                            #"Release Integrators" will create another file relative to the performance suite and the operators will fetch from that file the --cmsdriver option... for now just set the eventcontent since that is needed in order for things to run at all now...
                            stepOptions      = ""         ,
                            candleOptions    = ""         ,
                            profilers        = ""         ,
                            outputdir        = ""         ,
                            logfile          = None       ,
                            runonspare       = True       ,
                            bypasshlt        = False      ,
                            quicktest        = False      ,
                            unittest         = False      ,
                            dryrun           = False      ,
                            verbose          = True       ,
                            previousrel      = ""         ,
                            castordir        = self._CASTOR_DIR,
                            cores            = cmsCpuInfo.get_NumOfCores(), #Get Number of cpu cores on the machine from /proc/cpuinfo
                            cpu              = "1"        ) #Cpu core on which the suite is run:
    
        parser.add_option('-q', '--quiet'      , action="store_false", dest='verbose'   ,
            help = 'Output less information'                  )
        parser.add_option('-b', '--bypass-hlt' , action="store_true" , dest='bypasshlt' ,
            help = 'Bypass HLT root file as input to RAW2DIGI')
        parser.add_option('-n', '--notrunspare', action="store_false", dest='runonspare',
            help = 'Do not run cmsScimark on spare cores')        
        parser.add_option('-t', '--timesize'  , type='int'   , dest='TimeSizeEvents'  , metavar='<#EVENTS>'   ,
            help = 'specify the number of events for the TimeSize tests'                   )
        parser.add_option('-i', '--igprof'    , type='int'   , dest='IgProfEvents'    , metavar='<#EVENTS>'   ,
            help = 'specify the number of events for the IgProf tests'                     )
        parser.add_option('-v', '--valgrind'  , type='int'   , dest='ValgrindEvents'  , metavar='<#EVENTS>'   ,
            help = 'specify the number of events for the Valgrind tests'                   )
        parser.add_option('--cmsScimark'      , type='int'   , dest='cmsScimark'      , metavar=''            ,
            help = 'specify the number of times the cmsScimark benchmark is run before and after the performance suite on cpu1'         )
        parser.add_option('--cmsScimarkLarge' , type='int'   , dest='cmsScimarkLarge' , metavar=''            ,
            help = 'specify the number of times the cmsScimarkLarge benchmark is run before and after the performance suite on cpu1'    )
        parser.add_option('--cores'           , type='int', dest='cores'              , metavar='<CORES>'     ,
            help = 'specify the number of cores of the machine (can be used with 0 to stop cmsScimark from running on the other cores)' )        
        parser.add_option('-c', '--cmsdriver' , type='string', dest='cmsdriverOptions', metavar='<OPTION_STR>',
            help = 'specify special options to use with the cmsDriver.py commands (designed for integration build use'                  )        
        parser.add_option('-a', '--archive'   , type='string', dest='castordir'       , metavar='<DIR>'       ,
            help = 'specify the wanted CASTOR directory where to store the results tarball'                                             )
        parser.add_option('-L', '--logfile'   , type='string', dest='logfile'         , metavar='<FILE>'      ,
            help = 'file to store log output of the script'                                                                             )                
        parser.add_option('-o', '--output'    , type='string', dest='outputdir'       , metavar='<DIR>'       ,
            help = 'specify the directory where to store the output of the script'                                                      )        
        parser.add_option('-r', '--prevrel'   , type='string', dest='previousrel'     , metavar='<DIR>'       ,
            help = 'Top level dir of previous release for regression analysis'                                                          )        
        parser.add_option('--step'            , type='string', dest='stepOptions'     , metavar='<STEPS>'     ,
            help = 'specify the processing steps intended (instead of the default ones)'                                                )
        parser.add_option('--candle'          , type='string', dest='candleOptions'   , metavar='<CANDLES>'   ,
            help = 'specify the candle(s) to run (instead of all 7 default candles)'                                                    )
        parser.add_option('--cpu'             , type='string', dest='cpu'             , metavar='<CPU>'       ,
            help = 'specify the core on which to run the performance suite'                                                             )
    
    
        #####################
        #    
        # Developer options
        #
    
        devel  = opt.OptionGroup(parser, "Developer Options",
                                         "Caution: use these options at your own risk."
                                         "It is believed that some of them bite.\n")
    
        devel.add_option('-p', '--profile'  , type="str" , dest='profilers', metavar="<PROFILERS>" ,
            help = 'Profile codes to use for cmsRelvalInput' )
        devel.add_option('-f', '--false-run', action="store_true", dest='dryrun'   ,
            help = 'Dry run'                                                                                           )            
        devel.add_option('-d', '--debug'    , action='store_true', dest='debug'    ,
            help = 'Debug'                                                                                             )
        devel.add_option('--quicktest'      , action="store_true", dest='quicktest',
            help = 'Quick overwrite all the defaults to small numbers so that we can run a quick test of our chosing.' )  
        devel.add_option('--test'           , action="store_true", dest='unittest' ,
            help = 'Perform a simple test, overrides other options. Overrides verbosity and sets it to false.'         )            
    
        parser.add_option_group(devel)
        (options, args) = parser.parse_args(argslist)
    
    
        self._debug           = options.debug
        self._unittest        = options.unittest 
        self._verbose         = options.verbose
        self._dryrun          = options.dryrun    
        castordir        = options.castordir
        TimeSizeEvents   = options.TimeSizeEvents
        IgProfEvents     = options.IgProfEvents
        ValgrindEvents   = options.ValgrindEvents
        cmsScimark       = options.cmsScimark
        cmsScimarkLarge  = options.cmsScimarkLarge
        cmsdriverOptions = options.cmsdriverOptions
        stepOptions      = options.stepOptions
        quicktest        = options.quicktest
        candleoption     = options.candleOptions
        runonspare       = options.runonspare
        profilers        = options.profilers.strip()
        cpu              = options.cpu.strip()
        bypasshlt        = options.bypasshlt
        cores            = options.cores
        logfile          = options.logfile
        prevrel          = options.previousrel
        outputdir        = options.outputdir
    
        #################
        # Check logfile option
        #
        if not logfile == None:
            logfile = os.path.abspath(logfile)
            logdir = os.path.dirname(logfile)
            if not os.path.exists(logdir):
                parser.error("Directory to output logfile does not exist")
                sys.exit()
            logfile = os.path.abspath(logfile)
    
        #############
        # Check step Options
        #
        if "GEN,SIM" in stepOptions:
            print "WARNING: Please use GEN-SIM with a hypen not a \",\"!"
            
        if not stepOptions == "":
            #Wrapping the options with "" for the cmsSimPyRelVal.pl until .py developed
            stepOptions='"--usersteps=%s"' % (stepOptions)        
    
        ###############
        # Check profile option
        #
        isnumreg = re.compile("^-?[0-9]*$")
        found    = isnumreg.search(profilers)
        if not found :
            parser.error("profile codes option contains non-numbers")
            sys.exit()
    
        ###############
        # Check output directory option
        #
        if outputdir == "":
            outputdir = os.getcwd()
        else:
            outputdir = os.path.abspath(outputdir)
    
        if not os.path.isdir(outputdir):
            parser.error("%s is not a valid output directory" % outputdir)
            sys.exit()
            
        ################
        # Check cpu option
        # 
        numetcomreg = re.compile("^[0-9,]*")
        if not numetcomreg.search(cpu):
            parser.error("cpu option needs to be a comma separted list of ints or a single int")
            sys.exit()
    
        cpustr = cpu
        cpu = []
        if "," in cpustr:
            cpu = map(lambda x: int(x),cpustr.split(","))
        else:
            cpu = [ int(cpustr)  ]
    
        ################
        # Check previous release directory
        #
        if not prevrel == "":
            prevrel = os.path.abspath(prevrel)
            if not os.path.exists(prevrel):
                print "ERROR: Previous release dir %s could not be found" % prevrel
                sys.exit()
    
        #############
        # Setup quicktest option
        #
        if quicktest:
            TimeSizeEvents = 1
            IgProfEvents = 1
            ValgrindEvents = 0
            cmsScimark = 1
            cmsScimarkLarge = 1
    
        #############
        # Setup unit test option
        #
        if self._unittest:
            self._verbose = False
            if candleoption == "":
                candleoption = "MinBias"
            if stepOptions == "":
                stepOptions = "GEN-SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI-RECO"
            cmsScimark      = 0
            cmsScimarkLarge = 0
            ValgrindEvents  = 0
            IgProfEvents    = 0
            TimeSizeEvents  = 1
    
        #############
        # Setup cmsdriver option
        #
        if not cmsdriverOptions == "":
            cmsdriverOptions = "--cmsdriver=" + cmsdriverOptions        
            #Wrapping the options with "" for the cmsSimPyRelVal.pl until .py developed
            cmsdriverOptions= '"%s"' % (cmsdriverOptions)
            
        #############
        # Setup candle option
        #
        isAllCandles = candleoption == ""
        candles = {}
        if isAllCandles:
            candles = Candles
        else:
            candles = candleoption.split(",")
    
        return (castordir       ,
                TimeSizeEvents  ,
                IgProfEvents    ,
                ValgrindEvents  ,
                cmsScimark      ,
                cmsScimarkLarge ,
                cmsdriverOptions,
                stepOptions     ,
                quicktest       ,
                profilers       ,
                cpu             ,
                cores           ,
                prevrel         ,
                isAllCandles    ,
                candles         ,
                bypasshlt       ,
                runonspare      ,
                outputdir       ,
                logfile         )
    
    #def usage(self):
    #    return __doc__
    
    ############
    # Run a list of commands using system
    # ! We should rewrite this not to use system (most cases it is unnecessary)
    def runCmdSet(self,cmd):
        exitstat = 0
        if len(cmd) <= 1:
            exitstat = self.runcmd(cmd)
            if self._verbose:
                self.printFlush(cmd)
        else:
            for subcmd in cmd:
                if self._verbose:
                    self.printFlush(subcmd)
            exitstat = self.runcmd(" && ".join(cmd))
        if self._verbose:
            self.printFlush(self.getDate())
        return exitstat
    
    #############
    # Print and flush a string (for output to a log file)
    #
    def printFlush(self,command):
        if self._verbose:
            self.logh.write(command + "\n")
            self.logh.flush()
    
    #############
    # Run a command and return the exit status
    #
    def runcmd(self,command):
        #Substitute popen with subprocess.Popen!
        process  = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        #os.popen(command)
        exitstat= process.wait()
        cmdout   = process.stdout.read()
        #exitstat = process.returncode
        if self._verbose:
            self.logh.write(cmdout)# + "\n") No need of extra \n!
            self.logh.flush()
        if exitstat == None:
            print "Something strange is going on! Exit code was None for command %s: check if it really ran!"%command
            exitstat=0
        return exitstat
    
    def getDate(self):
        return time.ctime()
    
    def printDate(self):
        self.logh.write(self.getDate() + "\n")
    
    #############
    # If the minbias root file does not exist for qcd profiling then run a cmsDriver command to create it
    #
    def getPrereqRoot(self,rootdir,rootfile,cmsdriverOptions=""):
        self.logh.write("WARNING: %s file required to run QCD profiling does not exist. Now running cmsDriver.py to get Required Minbias root file\n"   % (rootdir + "/" +rootfile))
    
        if not os.path.exists(rootdir):
            os.system("mkdir -p %s" % rootdir)
        if not self._debug:
            cmd = "cd %s ; cmsDriver.py MinBias_cfi -s GEN,SIM -n %s %s >& ../minbias_for_pileup_generate.log" % (rootdir,str(10),cmsdriverOptions)
            self.logh.write(cmd)
            os.system(cmd)
        if not os.path.exists(rootdir + "/" + rootfile):
            self.logh.write("ERROR: We can not run QCD profiling please create root file %s to run QCD profiling.\n" % (rootdir + "/" + rootfile))
    
    #############
    # Check if QCD will run and if so check the root file is there. If it is not create it.
    #
    def checkQcdConditions(self,candles,TimeSizeEvents,rootdir,rootfile,cmsdriverOptions=""):
        if TimeSizeEvents < MIN_REQ_TS_EVENTS :
            self.logh.write("WARNING: TimeSizeEvents is less than %s but QCD needs at least that to run. PILE-UP will be ignored\n" % MIN_REQ_TS_EVENTS)
            
            
        rootfilepath = rootdir + "/" + rootfile
        if not os.path.exists(rootfilepath):
            self.getPrereqRoot(rootdir,rootfile,cmsdriverOptions)
            if not os.path.exists(rootfilepath) and not self._debug:
                self.logh.write("ERROR: Could not create or find a rootfile %s with enough TimeSize events for QCD exiting...\n" % rootfilepath)
                sys.exit()
        else:
            self.logh.write("%s Root file for QCD exists. Good!!!\n" % (rootdir + "/" + rootfile))
        return candles
    
    #############
    # Make directory for a particular candle and profiler.
    # ! This is really unnecessary code and should be replaced with a os.mkdir() call
    def mkCandleDir(self,pfdir,candle,profiler):
        adir = os.path.join(pfdir,"%s_%s" % (candle,profiler))
        self.runcmd( "mkdir -p %s" % adir )
        if self._verbose:
            self.printDate()
        return adir
    
    #############
    # Copy root file from another candle's directory
    # ! Again this is messy. 

    def cprootfile(self,dir,candle,NumOfEvents,cmsdriverOptions=""):
        cmds = ("cd %s" % dir,
                "cp -pR ../%s_IgProf/%s_GEN,SIM.root ."  % (candle,CandFname[candle]))
        
        if self.runCmdSet(cmds):
            self.logh.write("Since there was no ../%s_IgProf/%s_GEN,SIM.root file it will be generated first\n"%(candle,CandFname[candle]))

            cmd = "cd %s ; cmsDriver.py %s -s GEN,SIM -n %s --fileout %s_GEN,SIM.root %s>& %s_GEN_SIM_for_valgrind.log" % (dir,KeywordToCfi[candle],str(NumOfEvents),candle,cmsdriverOptions,candle)

            self.printFlush(cmd)
            cmdout=os.popen3(cmd)[2].read()
            if cmdout:
                self.printFlush(cmdout)
            return cmdout
            
    #############
    # Display G4 cerr errors and CMSExceptions in the logfile
    #
    def displayErrors(self,file):
        try:
            for line in open(file,"r"):
                if "cerr" in line or "CMSException" in line:
                    self.logh.write("ERROR: %s\n" % line)
                    self.ERRORS += 1
        except OSError, detail:
            self.logh.write("WARNING: %s\n" % detail)
            self.ERRORS += 1        
        except IOError, detail:
            self.logh.write("WARNING: %s\n" % detail)
            self.ERRORS += 1
        
    ##############
    # Filter lines in the valgrind report that match GEN,SIM
    #
    def valFilterReport(self,dir):
        #cmds = ("cd %s" % dir,
        #        "grep -v \"step=GEN,SIM\" SimulationCandles_%s.txt > tmp" % (self.cmssw_version),
        #        "mv tmp SimulationCandles_%s.txt"                         % (self.cmssw_version))
        #FIXME:
        #Quick and dirty hack to have valgrind MemCheck run on 5 events on both GEN,SIM and DIGI in QCD_80_120, while removing the line for GEN,SIM for Callgrind
        InputFileName=os.path.join(dir,"SimulationCandles_%s.txt"%(self.cmssw_version))
        InputFile=open(InputFileName,"r")
        InputLines=InputFile.readlines()
        InputFile.close()
        Outputfile=open(InputFileName,"w")
        simRegxp=re.compile("step=GEN,SIM")
        digiRegxp=re.compile("step=DIGI")
        CallgrindRegxp=re.compile("ValgrindFCE")
        MemcheckRegxp=re.compile("Memcheck")
        NumEvtRegxp=re.compile("-n 1")#FIXME Either use the ValgrindEventNumber or do a more general match!
        for line in InputLines:
            if simRegxp.search(line) and CallgrindRegxp.search(line):
                continue
            elif simRegxp.search(line) and MemcheckRegxp.search(line):
                #Modify
                if NumEvtRegxp.search(line):
                    line=NumEvtRegxp.sub(r"-n 5",line)
                else:
                    print "The number of Memcheck event was not changed since the original number of Callgrind event was not 1!"
                Outputfile.write(line)
            elif digiRegxp.search(line) and MemcheckRegxp.search(line):
                #Modify
                if NumEvtRegxp.search(line):
                    line=NumEvtRegxp.sub(r"-n 5",line)
                else:
                    print "The number of Memcheck event was not changed since the original number of Callgrind event was not 1!"
                Outputfile.write(line)
            else:
                Outputfile.write(line)
        Outputfile.close()
            
        #self.runCmdSet(cmds)
    
    ##################
    # Run cmsScimark benchmarks a number of times
    #
    def benchmarks(self,cpu,pfdir,name,bencher,large=False):
        cmd = self.Commands[cpu][3]
        redirect = ""
        if large:
            redirect = " -large >& "    
        else:
            redirect = " >& "
    
        for i in range(bencher):
            command= cmd + redirect + os.path.join(pfdir,os.path.basename(name))        
            self.printFlush(command + " [%s/%s]" % (i+1,bencher))
            self.runcmd(command)
            self.logh.flush()
    
    ##################
    # This function is a wrapper around cmsRelvalreport
    # 
    def runCmsReport(self,cpu,dir,candle):
        cmd  = self.Commands[cpu][1]
        cmds = ("cd %s"                 % (dir),
                "%s -i SimulationCandles_%s.txt -t perfreport_tmp -R -P >& %s.log" % (cmd,self.cmssw_version,candle))
        exitstat = 0
        if not self._debug:
            exitstat = self.runCmdSet(cmds)
            
        if self._unittest and (not exitstat == 0):
            self.logh.write("ERROR: CMS Report returned a non-zero exit status \n")
            sys.exit(exitstat)
        else:
            return(exitstat) #To return the exit code of the cmsRelvalreport.py commands to the runPerfSuite function
    
    ##################
    # Test cmsDriver.py (parses the simcandles file, removing duplicate lines, and runs the cmsDriver part)
    #
    def testCmsDriver(self,cpu,dir,cmsver,candle):
        cmsdrvreg = re.compile("^cmsDriver.py")
        cmd  = self.Commands[cpu][0]
        noExit = True
        stepreg = re.compile("--step=([^ ]*)")
        previousCmdOnline = ""
        for line in open(os.path.join(dir,"SimulationCandles_%s.txt" % (cmsver))):
            if (not line.lstrip().startswith("#")) and not (line.isspace() or len(line) == 0): 
                cmdonline  = line.split("@@@",1)[0]
                if cmsdrvreg.search(cmdonline) and not previousCmdOnline == cmdonline:
                    stepbeingrun = "Unknown"
                    matches = stepreg.search(cmdonline)
                    if not matches == None:
                        stepbeingrun = matches.groups()[0]
                    if "PILEUP" in cmdonline:
                        stepbeingrun += "_PILEUP"
                    self.logh.write(cmdonline + "\n")
                    cmds = ("cd %s"      % (dir),
                            "%s  >& ../cmsdriver_unit_test_%s_%s.log"    % (cmdonline,candle,stepbeingrun))
                    if self._dryrun:
                        self.logh.write(cmds + "\n")
                    else:
                        out = self.runCmdSet(cmds)                    
                        if not out == None:
                            sig     = out >> 16    # Get the top 16 bits
                            xstatus = out & 0xffff # Mask out all bits except the first 16 
                            self.logh.write("FATAL ERROR: CMS Driver returned a non-zero exit status (which is %s) when running %s for candle %s. Signal interrupt was %s\n" % (xstatus,stepbeingrun,candle,sig))
                            sys.exit()
                previousCmdOnline = cmdonline
        
    ##############
    # Wrapper for cmsRelvalreportInput 
    # 
    def runCmsInput(self,cpu,dir,numevents,candle,cmsdrvopts,stepopt,profiles,bypasshlt):
    
        bypass = ""
        if bypasshlt:
            bypass = "--bypass-hlt"
        cmd = self.Commands[cpu][2]
        cmds=[]
        #for cpu in self.Commands.keys():
        #    cmds.append("cd %s"%(dir[cpu]))
        #    cmds.append("%s %s \"%s\" %s %s %s %s" % (cmd,
        #                                              numevents,
        #                                              candle,
        #                                              profiles,
        #                                              cmsdrvopts,
        #                                              stepopt,
        #                                              bypass))
        #print cmds
        cmds = ("cd %s"                    % (dir),
                "%s %s \"%s\" %s %s %s %s" % (cmd,
                                              numevents,
                                              candle,
                                              profiles,
                                              cmsdrvopts,
                                              stepopt,
                                              bypass))
        exitstat=0
        exitstat = self.runCmdSet(cmds)
        if self._unittest and (not exitstat == 0):
            self.logh.write("ERROR: CMS Report Input returned a non-zero exit status \n" )
        return exitstat
    ##############
    # Prepares the profiling directory and runs all the selected profiles (if this is not a unit test)
    #
    def simpleGenReport(self,cpus,perfdir,NumEvents,candles,cmsdriverOptions,stepOptions,Name,profilers,bypasshlt):
        valgrind = Name == "Valgrind"
    
        profCodes = {"TimeSize" : "0123",
                     "IgProf"   : "4567",
                     "Valgrind" : "89",
                     None       : "-1"} 
    
        profiles = profCodes[Name]
        if not profilers == "":
            profiles = profilers        
        #adir={}
        RelvalreportExitCode=0
        for cpu in cpus:
            pfdir = perfdir
            if len(cpus) > 1:
                pfdir = os.path.join(perfdir,"cpu_%s" % cpu)
            for candle in candles:
                #adir[cpu]=self.mkCandleDir(pfdir,candle,Name)
                adir=self.mkCandleDir(pfdir,candle,Name)
                if valgrind:
                    if candle == "SingleMuMinusPt10" : 
                        self.logh.write("Valgrind tests **GEN,SIM ONLY** on %s candle\n" % candle    )
                    else:
                        self.logh.write("Valgrind tests **SKIPPING GEN,SIM** on %s candle\n" % candle)

                        #self.cprootfile(adir,candle,NumEvents,cmsdriverOptions[13:-1])#Nasty hack to propagate cmsdriverOptions to potential cmsDriver.py commands to create necessary root files...              

    
                if self._unittest:
                    # Run cmsDriver.py
                    self.runCmsInput(cpu,adir,NumEvents,candle,cmsdriverOptions,stepOptions,profiles,bypasshlt)
                    self.testCmsDriver(cpu,adir,candle)
                else:
                    self.runCmsInput(cpu,adir,NumEvents,candle,cmsdriverOptions,stepOptions,profiles,bypasshlt)            
                    if valgrind and candle == "QCD_80_120":
                        self.valFilterReport(adir)
                    ExitCode=self.runCmsReport(cpu,adir,candle)
                    print "Individual Relvalreport.py ExitCode %s"%ExitCode
                    RelvalreportExitCode=RelvalreportExitCode+ExitCode
                    print "Summed Relvalreport.py ExitCode %s"%RelvalreportExitCode
                    #proflogs = []
                    #Change the log testing to look for G4 cerr but also for CMSException
                    #Also look in the main cmsRelvalreport log (not the TimingReport only)
                    #That contains all other information.
                    #if   Name == "TimeSize":
                    #    proflogs = [ "TimingReport" ]
                    #elif Name == "Valgrind":
                    #    pass
                    #elif Name == "IgProf":
                    #    pass
                    #
                    #for proflog in proflogs:
                    #With the change from 2>1&|tee to >& to preserve exit codes, we need now to check all logs...
                    #less nice... we might want to do this externally so that in post-processing its a re-usable tool
                    globpath = os.path.join(adir,"*.log") #"%s.log"%candle)
                    self.logh.write("Looking for logs that match %s\n" % globpath)
                    logs     = glob.glob(globpath)
                    for log in logs:
                        self.logh.write("Found log %s\n" % log)
                        self.displayErrors(log)
        print "Returned cumulative RelvalreportExitCode is %s"%RelvalreportExitCode
        return RelvalreportExitCode
    
    ############
    # Runs benchmarking, cpu spinlocks on spare cores and profiles selected candles
    #
    #FIXME:
    #Could redesign interface of functions to use keyword arguments:
    #def runPerfSuite(**opts):
    #then instead of using castordir variable, would use opts['castordir'] etc    
    def runPerfSuite(self,
                     castordir        = "/castor/cern.ch/cms/store/relval/performance/",
                     TimeSizeEvents   = 100        ,
                     IgProfEvents     = 5          ,
                     ValgrindEvents   = 1          ,
                     cmsScimark       = 10         ,
                     cmsScimarkLarge  = 10         ,
                     cmsdriverOptions = ""         ,#Could use directly cmsRelValCmd.get_Options()
                     stepOptions      = ""         ,
                     quicktest        = False      ,
                     profilers        = ""         ,
                     cpus             = [1]        ,
                     cores            = 4          ,#Could use directly cmsCpuInfo.get_NumOfCores()
                     prevrel          = ""         ,
                     isAllCandles     = False      ,
                     candles          = Candles    ,
                     bypasshlt        = False      ,
                     runonspare       = True       ,
                     perfsuitedir     = os.getcwd(),
                     logfile          = os.path.join(os.getcwd(),"cmsPerfSuite.log")):
        #Set up a variable for the FinalExitCode to be used as the sum of exit codes:
        FinalExitCode=0
        #Print a time stamp at the beginning:
    
        if not logfile == None:
            try:
                self.logh = open(logfile,"a")
            except (OSError, IOError), detail:
                self.logh.write(detail + "\n")
    
        try:        
            if not prevrel == "":
                self.logh.write("Production of regression information has been requested with release directory %s" % prevrel)
            if not cmsdriverOptions == "":
                self.logh.write("Running cmsDriver.py with the special user defined options: %s\n" % cmsdriverOptions)
                #Attach the full option synthax for cmsRelvalreportInput.py:
                cmsdriverOptionsRelvalInput="--cmsdriver="+cmsdriverOptions
                #FIXME: should import cmsRelvalreportInput.py and avoid these issues...
            if not stepOptions == "":
                self.logh.write("Running user defined steps only: %s\n" % stepOptions)
                #Attach the full option synthax for cmsRelvalreportInput.py:
                setpOptionsRelvalInput="--usersteps="+stepOptions
                #FIXME: should import cmsRelvalreportInput.py and avoid these issues...
            if bypasshlt:
                #Attach the full option synthax for cmsRelvalreportInput.py:
                bypasshltRelvalInput="--bypass-hlt"
                #FIXME: should import cmsRelvalreportInput.py and avoid these issues...
            if not len(candles) == len(Candles):
                self.logh.write("Running only %s candle, instead of the whole suite\n" % str(candles))
            
            self.logh.write("This machine ( %s ) is assumed to have %s cores, and the suite will be run on cpu %s\n" %(self.host,cores,cpus))
            path=os.path.abspath(".")
            self.logh.write("Performance Suite started running at %s on %s in directory %s, run by user %s\n" % (self.getDate(),self.host,path,self.user))
            showtags=os.popen4("showtags -r")[1].read()
            self.logh.write(showtags) # + "\n") No need for extra \n!
    
            #For the log:
            if self._verbose:
                self.logh.write("The performance suite results tarball will be stored in CASTOR at %s\n" % self._CASTOR_DIR)
                self.logh.write("%s TimeSize events\n" % TimeSizeEvents)
                self.logh.write("%s IgProf events\n"   % IgProfEvents)
                self.logh.write("%s Valgrind events\n" % ValgrindEvents)
                self.logh.write("%s cmsScimark benchmarks before starting the tests\n"      % cmsScimark)
                self.logh.write("%s cmsScimarkLarge benchmarks before starting the tests\n" % cmsScimarkLarge)
    
            #Actual script actions!
            #Will have to fix the issue with the matplotlib pie-charts:
            #Used to source /afs/cern.ch/user/d/dpiparo/w0/perfreport2.1installation/share/perfreport/init_matplotlib.sh
            #Need an alternative in the release
    
    
    
            if len(cpus) > 1:
                for cpu in cpus:
                    cpupath = os.path.join(perfsuitedir,"cpu_%s" % cpu)
                    if not os.path.exists(cpupath):
                        os.mkdir(cpupath)
    
    
            self.Commands = {}
            AllScripts = self.Scripts + self.AuxiliaryScripts
    
            for cpu in cpus:
                self.Commands[cpu] = []
    
            self.logh.write("Full path of all the scripts used in this run of the Performance Suite:\n")
            for script in AllScripts:
                which="which " + script
    
                #Logging the actual version of cmsDriver.py, cmsRelvalreport.py, cmsSimPyRelVal.pl
                whichstdout=os.popen4(which)[1].read()
                self.logh.write(whichstdout) # + "\n") No need of the extra \n!
                if script in self.Scripts:
                    for cpu in cpus:
                        command="taskset -c %s %s" % (cpu,script)
                        self.Commands[cpu].append(command)
    
            #First submit the cmsScimark benchmarks on the unused cores:
            scimark = ""
            scimarklarge = ""
            if not self._unittest:
                for core in range(cores):
                    if (not core in cpus) and runonspare:
                        self.logh.write("Submitting cmsScimarkLaunch.csh to run on core cpu "+str(core) + "\n")
                        subcmd = "cd %s ; cmsScimarkLaunch.csh %s" % (perfsuitedir, str(core))            
                        command="taskset -c %s sh -c \"%s\" &" % (str(core), subcmd)
                        self.logh.write(command + "\n")
    
                        #cmsScimarkLaunch.csh is an infinite loop to spawn cmsScimark2 on the other
                        #cpus so it makes no sense to try reading its stdout/err 
                        os.popen4(command)

            self.logh.flush()
    
            #dont do benchmarking if in debug mode... saves time
            benching = not self._debug
            if benching and not self._unittest:
                #Submit the cmsScimark benchmarks on the cpu where the suite will be run:
                for cpu in cpus:
                    scimark      = open(os.path.join(perfsuitedir,"cmsScimark2.log")      ,"w")        
                    scimarklarge = open(os.path.join(perfsuitedir,"cmsScimark2_large.log"),"w")
                    if cmsScimark > 0:
                        self.logh.write("Starting with %s cmsScimark on cpu%s\n"       % (cmsScimark,cpu))
                        self.benchmarks(cpu,perfsuitedir,scimark.name,cmsScimark)
    
                    if cmsScimarkLarge > 0:
                        self.logh.write("Following with %s cmsScimarkLarge on cpu%s\n" % (cmsScimarkLarge,cpu))
                        self.benchmarks(cpu,perfsuitedir,scimarklarge.name,cmsScimarkLarge)
    
            if not profilers == "":
                # which profile sets should we go into if custom profiles have been selected
                runTime     = reduce(lambda x,y: x or y, map(lambda x: x in profilers, ["0", "1", "2", "3"]))
                runIgProf   = reduce(lambda x,y: x or y, map(lambda x: x in profilers, ["4", "5", "6", "7"]))
                runValgrind = reduce(lambda x,y: x or y, map(lambda x: x in profilers, ["8", "9"]))
                if not runTime:
                    TimeSizeEvents = 0
                if not runIgProf:
                    IgProfEvents   = 0
                if not runValgrind:
                    ValgrindEvents = 0
    
            qcdWillRun = (not isAllCandles) and "QCD_80_120" in candles 
            if qcdWillRun:
                candles = self.checkQcdConditions(candles,
                                             TimeSizeEvents,
                                             os.path.join(perfsuitedir,"%s_%s" % ("MinBias","TimeSize")),
                                             "%s_cfi_GEN_SIM.root" % "MinBias",cmsdriverOptions[13:-1])#Really nasty hack to pass the cmsdriverOptions to the various checkers that could run cmsDriver.py to create needed input files
    
            #TimeSize tests:
            if TimeSizeEvents > 0:
                self.logh.write("Launching the TimeSize tests (TimingReport, TimeReport, SimpleMemoryCheck, EdmSize) with %s events each\n" % TimeSizeEvents)
                self.printDate()
                self.logh.flush()
                ReportExit=self.simpleGenReport(cpus,perfsuitedir,TimeSizeEvents,candles,cmsdriverOptions,stepOptions,"TimeSize",profilers,bypasshlt)
                FinalExitCode=FinalExitCode+ReportExit
    
            #IgProf tests:
            if IgProfEvents > 0:
                self.logh.write("Launching the IgProf tests (IgProfPerf, IgProfMemTotal, IgProfMemLive, IgProfMemAnalyse) with %s events each\n" % IgProfEvents)
                self.printDate()
                self.logh.flush()
                IgCandles = candles
                #By default run IgProf only on QCD_80_120 candle
                if isAllCandles:
                    IgCandles = [ "QCD_80_120" ]
                ReportExit=self.simpleGenReport(cpus,perfsuitedir,IgProfEvents,IgCandles,cmsdriverOptions,stepOptions,"IgProf",profilers,bypasshlt)
                FinalExitCode=FinalExitCode+ReportExit
            #Stopping all cmsScimark jobs and analysing automatically the logfiles
            #No need to waste CPU while the load does not affect Valgrind measurements!
            self.logh.write("Stopping all cmsScimark jobs now\n")
            subcmd = "cd %s ; %s" % (perfsuitedir,self.AuxiliaryScripts[2])
            stopcmd = "sh -c \"%s\"" % subcmd
            self.printFlush(stopcmd)
            #os.popen(stopcmd)
            self.printFlush(os.popen4(stopcmd)[1].read())
            
            #Valgrind tests:
            if ValgrindEvents > 0:
                #FIXME
                       
                #1-Could launch different tests on different cores to parallelize:
                #  a-Callgrind on QCD_80_120 on one core (unprofiled GEN,SIM, profile DIGI)
                #  b-Callgrind on SingleMu on another core
                #  c-Memcheck on QCD_80_120 on another core
                #  d-Memcheck on SingleMu on another core
                #  This could become a problem if one wants to launch the whole suite
                #  as a separate thread on a certain core: catch this in the options.
                #  if the --cpu is specified then no "parallelizing" of the valgrind part, this should be enough
                #  Cannot be done! --cpu 1 is the default... to either one catches this in the parsing of the arguments, or a new argument --valgringThreading is added, or we assume valgrind is never run when threading the whole suite... for now let's do this last option:
                
                self.logh.write("Launching the Valgrind tests (callgrind_FCE, memcheck) with %s events each\n" % ValgrindEvents)
                self.printDate()
                self.logh.flush()
                valCandles = candles
    
                if isAllCandles:
                    cmds=[]
                    #By default run Valgrind only on QCD_80_120, skipping SIM step since it would take forever (and do SIM step on SingleMu)
                    valCandles = [ "QCD_80_120" ]
    
                #Besides always run, only once the GEN,SIM step on SingleMu:
                valCandles.append("SingleMuMinusPt10")
                #In the user-defined candles a different behavior: do Valgrind for all specified candles (usually it will only be 1)
                #usercandles=candleoption.split(",")
                ReportExit=self.simpleGenReport(cpus,perfsuitedir,ValgrindEvents,valCandles,cmsdriverOptions,stepOptions,"Valgrind",profilers,bypasshlt)
                FinalExitCode=FinalExitCode+ReportExit
            if benching and not self._unittest:
                #Ending the performance suite with the cmsScimark benchmarks again:
                for cpu in cpus:
                    if cmsScimark > 0:
                        self.logh.write("Ending with %s cmsScimark on cpu%s\n"         % (cmsScimark,cpu))
                        self.benchmarks(cpu,perfsuitedir,scimark.name,cmsScimark)
    
                    if cmsScimarkLarge > 0:
                        self.logh.write("Following with %s cmsScimarkLarge on cpu%s\n" % (cmsScimarkLarge,cpu))
                        self.benchmarks(cpu,perfsuitedir,scimarklarge.name,cmsScimarkLarge)
    
            if not prevrel == "":
                crr.regressReports(prevrel,os.path.abspath(perfsuitedir),oldRelName = getVerFromLog(prevrel),newRelName=self.cmssw_version)
    
            #Create a tarball of the work directory
            #Adding the str(stepOptions to distinguish the tarballs for 1 release (GEN->DIGI, L1->RECO will be run in parallel)
            TarFile = "%s_%s_%s_%s.tar" % (self.cmssw_version, str(stepOptions), self.host, self.user)
            AbsTarFile = os.path.join(perfsuitedir,TarFile)
            tarcmd  = "tar -cf %s %s; gzip %s" % (AbsTarFile,os.path.join(perfsuitedir,"*"),AbsTarFile)
            self.printFlush(tarcmd)
            self.printFlush(os.popen3(tarcmd)[2].read()) #Using popen3 to get only stderr we don't want the whole stdout of tar!
    
            #Archive it on CASTOR
            castorcmd="rfcp %s.gz %s.gz" % (AbsTarFile,os.path.join(self._CASTOR_DIR,TarFile))
            self.printFlush(castorcmd)
            castorcmdstderr=os.popen3(castorcmd)[2].read()
            #Checking the stderr of the rfcp command to copy the tarball.gz on CASTOR:
            if castorcmdstderr:
                #If it failed print the stderr message to the log and tell the user the tarball.gz is kept in the working directory
                self.printFlush(castorcmdstderr)
                self.printFlush("Since the CASTOR archiving for the tarball failed the file %s is kept in directory %s"%(TarFile, perfsuitedir))
            else:
                #If it was successful then remove the tarball.gz from the working directory:
                TarGzipFile=TarFile+".gz"
                self.printFlush("Successfully archived the tarball %s in CASTOR!\nDeleting the local copy of the tarball"%(TarGzipFile))
                AbsTarGzipFile=AbsTarFile+".gz"
                rmtarballcmd="rm -Rf %s"%(AbsTarGzipFile)
                self.printFlush(rmtarballcmd)
                self.printFlush(os.popen4(rmtarballcmd)[1].read())
                
            #End of script actions!
    
            #Print a time stamp at the end:
            date=time.ctime(time.time())
            self.logh.write("Performance Suite finished running at %s on %s in directory %s\n" % (date,self.host,path))
            if self.ERRORS == 0:
                self.logh.write("There were no errors detected in any of the log files!\n")
            else:
                self.logh.write("ERROR: There were %s errors detected in the log files, please revise!\n" % self.ERRORS)
                #print "No exit code test"
                #sys.exit(1)
        except exceptions.Exception, detail:
            self.logh.write(str(detail) + "\n")
            self.logh.flush()
            if not self.logh.isatty():
                self.logh.close()
            raise
        sys.exit(FinalExitCode)
    
def main(argv=[__name__]): #argv is a list of arguments.
                     #Valid ways to call main with arguments:
                     #main(["--cmsScimark",10])
                     #main(["-t100"]) #With the caveat that the options.timeSize will be of type string... so should avoid using this!
                     #main(["-timeSize,100])
                     #Invalid ways:
                     #main(["One string with all options"])
    #Let's instatiate the class:
    suite=PerfSuite()
    print suite                      
    #Let's check the command line arguments
    #(castordir       ,
    # TimeSizeEvents  ,
    # IgProfEvents    ,
    # ValgrindEvents  ,
    # cmsScimark      ,
    # cmsScimarkLarge ,
    # cmsdriverOptions,
    # stepOptions     ,
    # quicktest       ,
    # profilers       ,
    # cpus            ,
    # cores           ,
    # prevrel         ,
    # isAllCandles    ,
    # candles         ,
    # bypasshlt       ,
    # runonspare      ,
    # outputdir       ,
    # logfile         ) = suite.optionParse(argv)
    PerfSuiteArgs={}
    (PerfSuiteArgs['castordir'],
     PerfSuiteArgs['TimeSizeEvents'],
     PerfSuiteArgs['IgProfEvents'],    
     PerfSuiteArgs['ValgrindEvents'],  
     PerfSuiteArgs['cmsScimark'],      
     PerfSuiteArgs['cmsScimarkLarge'], 
     PerfSuiteArgs['cmsdriverOptions'],
     PerfSuiteArgs['stepOptions'],     
     PerfSuiteArgs['quicktest'],       
     PerfSuiteArgs['profilers'],       
     PerfSuiteArgs['cpus'],            
     PerfSuiteArgs['cores'],           
     PerfSuiteArgs['prevrel'],         
     PerfSuiteArgs['isAllCandles'],    
     PerfSuiteArgs['candles'],         
     PerfSuiteArgs['bypasshlt'],       
     PerfSuiteArgs['runonspare'],      
     PerfSuiteArgs['perfsuitedir'],    
     PerfSuiteArgs['logfile'],
     ) = suite.optionParse(argv)
    print "Initial PerfSuite Arguments:"
    for key in PerfSuiteArgs.keys():
        print key,PerfSuiteArgs[key]
    print PerfSuiteArgs
    #Handle in here the case of multiple cores and the loading of cores with cmsScimark:
    if len(PerfSuiteArgs['cpus']) > 1:
        print "More than 1 cpu: threading the Performance Suite!"
        outputdir=PerfSuiteArgs['perfsuitedir']
        runonspare=PerfSuiteArgs['runonspare'] #Save the original value of runonspare for cmsScimark stuff
        cpus=PerfSuiteArgs['cpus']
        if runonspare:
            for core in range(PerfSuiteArgs['cores']):
                cmsScimarkLaunch_pslist={}
                if (core not in cpus):
                    #self.logh.write("Submitting cmsScimarkLaunch.csh to run on core cpu "+str(core) + "\n")
                    print "Submitting cmsScimarkLaunch.csh to run on core cpu "+str(core)+"\n"
                    subcmd = "cd %s ; cmsScimarkLaunch.csh %s" % (outputdir, str(core))            
                    command="taskset -c %s sh -c \"%s\" &" % (str(core), subcmd)
                    #self.logh.write(command + "\n")
                    print command + "\n"
                    
                    #cmsScimarkLaunch.csh is an infinite loop to spawn cmsScimark2 on the other
                    #cpus so it makes no sense to try reading its stdout/err
                    cmsScimarkLaunch_pslist[core]=subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    print "Spawned %s \n with PID %s"%(command,cmsScimarkLaunch_pslist[core].pid)
        PerfSuiteArgs['runonspare']=False #Set it to false to avoid cmsScimark being spawned by each thread
        logfile=PerfSuiteArgs['logfile']
        suitethread={}
        for cpu in cpus:
            #Make arguments "threaded" by setting for each instance of the suite:
            #1-A different output (sub)directory
            #2-Only 1 core on which to run
            #3-Automatically have a logfile... otherwise stdout is lost?
            #To be done:[3-A flag for Valgrind not to "thread" itself onto the other cores..]
            cpudir = os.path.join(outputdir,"cpu_%s" % cpu)
            if not os.path.exists(cpudir):
                os.mkdir(cpudir)
            PerfSuiteArgs['perfsuitedir']=cpudir
            PerfSuiteArgs['cpus']=[cpu]  #Keeping the name cpus for now FIXME: change it to cpu in the whole code
            if PerfSuiteArgs['logfile']:
                PerfSuiteArgs['logfile']=os.path.join(cpudir,os.path.basename(PerfSuiteArgs['logfile']))
            else:
                PerfSuiteArgs['logfile']=os.path.join(cpudir,"cmsPerfSuiteThread.log")
            #Now spawn the thread with:
            suitethread[cpu]=PerfThread(**PerfSuiteArgs)
            print suitethread[cpu]
            print "Launching PerfSuite thread on cpu%s"%cpu
            #print "With arguments:"
            #print PerfSuiteArgs
            suitethread[cpu].start()
        #print suitethread
        #FIXME Do somthing to kill the cmsScimarks after all threads are done...
        #Fix the cmsScimarkLAunch and Stop into being python and threaded themselves?
        #while True:
        #    threadsdone=0
        #    for cpu in suitethread.keys():
        #        if suitethread[cpu].isAlive:
        #            print "%s is ALIVE!"%suitethread[cpu]
        #            sys.stdout.flush()
        #            continue
        #        else:
        #            threadsdone=threadsdone+1
        #            print threadsdone
        #            sys.stdout.flush()
        #            
        #    if threadsdone == len(suitethread.keys()):
        #        for core in cmsScimarkLaunch_pslist.keys():
        #            kill=subprocess.Popen("kill %s"%cmsScimarkLaunch_pslist[core].pid,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        #            print kill.stdout.read()
        #        sys.exit()
        #    time.sleep(10)

    else: #No threading, just run the performance suite on the cpu core selected
        suite.runPerfSuite(**PerfSuiteArgs)
    
if __name__ == "__main__":
    main(sys.argv)
