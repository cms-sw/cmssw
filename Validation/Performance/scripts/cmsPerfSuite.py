#!/usr/bin/env python
import os, time, sys, re, glob, exceptions
import optparse as opt
import cmsRelRegress as crr
from cmsPerfCommons import Candles, KeywordToCfi, CandFname, cmsDriverPileUpOption, getVerFromLog
import cmsRelValCmd,cmsCpuInfo
import threading #Needed in threading use for Valgrind
import subprocess #Nicer subprocess management than os.popen
import datetime #Used to time the running of the performance suite
import pickle #Used to dump the running timing information

#Redefine _cleanup() function not to poll active processes
#[This is necessary to avoid issues when threading]
#So let's have it do nothing:
def _cleanup():
   pass
#Override the function in subprocess
subprocess._cleanup=_cleanup

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

class PerfSuiteTimer:
   """A class defining timing objects to time the running of the various parts of the performance suite. The class depends on module datetime."""
   def __init__(self,start=None):
      """Initialize the start time and set the end time to some indefinite time in the future"""
      self.start = start
      self.end = datetime.datetime.max
      self.duration = self.start - self.end

   #Setters:
   def set_start(self,start=None):
      self.start = start
   def set_end(self,end=None):
      #print "Setting end time to %s"%end.ctime()
      self.end = end
      self.duration = self.end - self.start
   #Getters
   def get_start(self):
      """Return the start time in ctime timestamp format"""
      return self.start.ctime()
   def get_end(self):
      """Return the end time in ctime timestamp format"""
      return self.end.ctime()
   def get_duration(self):
      """Return the duration between start and end as a dictionary with keys 'hours', 'minutes', 'seconds' to express the total duration in the favourite (most appropriate) unit. The function returns truncated integers."""
      self.duration_seconds = self.duration.days*86400 + self.duration.seconds
      self.duration_minutes = self.duration_seconds/60
      self.duration_hours = self.duration_seconds/3600
      return {'hours':self.duration_hours, 'minutes':self.duration_minutes, 'seconds':self.duration_seconds}

class PerfSuite:
    def __init__(self):
        
        self.ERRORS = 0
        #Swtiching from CASTOR to EOS (using xrdcp instead of rfcp and  root://eoscms//eos/ instead of /castor/cern.ch/
        #NOT YET!
        #FIX ME... do the migration to EOS eventually, taking care of PerFDB implications for tarball location!
        self._CASTOR_DIR = "/castor/cern.ch/cms/store/relval/performance/"
        self._dryrun   = False
        self._debug    = False
        self._unittest = False
        self._noexec   = False
        self._verbose  = True
        self.logh = sys.stdout
    
        #Get some environment variables to use
        try:
            self.cmssw_arch   = os.environ["SCRAM_ARCH"]
            self.cmssw_version= os.environ["CMSSW_VERSION"]
            self.host         = os.environ["HOST"]
            self.user              = os.environ["USER"]
        except KeyError:
            self.logh.write('Error: An environment variable either SCRAM_ARCH, CMSSW_VERSION, HOST or USER is not available.\n')
            self.logh.write('       Please run eval `scramv1 runtime -csh` to set your environment variables\n')
            self.logh.flush()
            sys.exit()
           
        #Scripts used by the suite:
        self.Scripts         =["cmsDriver.py","cmsRelvalreport.py","cmsRelvalreportInput.py","cmsScimark2"]
        self.AuxiliaryScripts=["cmsScimarkLaunch.csh","cmsScimarkParser.py","cmsScimarkStop.py"]

        
    #Threading the execution of IgProf, Memcheck and Callgrind using the same model used to thread the whole performance suite:
    #1-Define a class simpleGenReportThread() that has relevant methods needed to handle PerfTest()
    #2-Instantiate one with the necessary arguments to run simpleGenReport on core N
    #3-Execute its "run" method by starting the thread
    #Simplest way maybe is to keep 2 global lists:
    #AvailableCores
    #TestsToDo
    #PerfSuite will fill the TestsToDo list with dictionaries, to be used as keyword arguments to instantiate a relevant thread.
    #Once all the TestsToDo are "scheduled" into the list (FirstInLastOut buffer since we use pop()) PerfSuite will look into the
    #AvailableCores list and start popping cores onto which to instantiate the relevant threads, then it will start the thread,
    #appending it to the activePerfTestThread{},a dictionary with core as key and thread object as value, to facilitate bookkeeping.
    #An infinite loop will take care of checking for AvailableCores as long as there are TestsToDo and keep submitting.
    #In the same loop the activePerfTestThread{} will be checked for finished threads and it will re-append the relevant cpu back
    #to the AvailableCores list.
    #In the same loop a check for the case of all cores being back into AvailableCores with no more TestsToDo will break the infinite loop
    #and declare the end of all tests.As else to this if a sleep statement of 5 seconds will delay the repetition of the loop.

    def createIgVolume(self):
       igcommand = '/afs/cern.ch/cms/sdt/internal/scripts/requestPerfIgprofSpace.py --version ' + self.cmssw_version + ' --platform ' + self.cmssw_arch
       subprocess.Popen(igcommand,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)


    class simpleGenReportThread(threading.Thread):
       def __init__(self,cpu,perfsuiteinstance,**simpleGenReportArgs): #Passing around the perfsuite object to be able to access simpleGenReport
          self.cpu=cpu
          self.simpleGenReportArgs=simpleGenReportArgs
          self.perfsuiteinstance=perfsuiteinstance
          threading.Thread.__init__(self)
       def run(self):
          self.PerfTest=self.perfsuiteinstance.PerfTest(self.cpu,self.perfsuiteinstance,**(self.simpleGenReportArgs))
          self.PerfTest.runPerfTest()
    
    class PerfTest:
       def __init__(self,cpu,perfsuiteinstance,**simpleGenReportArgs):
          self.cpu=cpu
          self.simpleGenReportArgs=simpleGenReportArgs
          self.perfsuiteinstance=perfsuiteinstance
       def runPerfTest(self):
#          self.PerfTestTotalTimer=PerfSuiteTimer(start=datetime.datetime.now()) #Start the TimeSize timer
#          TimerInfo.update({self.simpleGenReportArgs['Name']:{'TotalTime':self.PerfTestTotalTimer}}) #Add the TimeSize timer to the dictionary  
          if "--pileup" in self.simpleGenReportArgs['cmsdriverOptions']:
             self.perfsuiteinstance.logh.write("Launching the PILE UP %s tests on cpu %s with %s events each\n"%(self.simpleGenReportArgs['Name'],self.cpu,self.simpleGenReportArgs['NumEvents']))
             self.PerfTestPUTimer=PerfSuiteTimer(start=datetime.datetime.now()) #Start the TimeSize timer
             TimerInfo[self.simpleGenReportArgs['Name']].update({'PileUpTime':self.PerfTestPUTimer}) #Add the TimeSize timer to the dictionary
             
          else:
             self.perfsuiteinstance.logh.write("Launching the %s tests on cpu %s with %s events each\n"%(self.simpleGenReportArgs['Name'],self.cpu,self.simpleGenReportArgs['NumEvents']))
             self.PerfTestTimer=PerfSuiteTimer(start=datetime.datetime.now()) #Start the TimeSize timer
             TimerInfo[self.simpleGenReportArgs['Name']].update({'NoPileUpTime':self.PerfTestTimer}) #Add the TimeSize timer to the dictionary
          self.perfsuiteinstance.logh.flush()
          #Cut and paste in bulk, should see if this works...
          self.perfsuiteinstance.printDate()
          self.perfsuiteinstance.logh.flush()
          self.exitcode=self.perfsuiteinstance.simpleGenReport([self.cpu],**(self.simpleGenReportArgs)) #Returning ReportExit code
          #Stop the timers on the threaded PileUp and NoPileUp tests:
          if "--pileup" in self.simpleGenReportArgs['cmsdriverOptions']:
             self.PerfTestPUTimer.set_end(datetime.datetime.now())
          else:
             self.PerfTestTimer.set_end(datetime.datetime.now())
          return self.exitcode
       
    #Options handling
    def optionParse(self,argslist=None):
        parser = opt.OptionParser(usage='''./cmsPerfSuite.py [options]
           
    Examples:
    
    cmsPerfSuite.py --step GEN-HLT -t 5 -i 2 -c 1 -m 5 --RunTimeSize MinBias,TTbar --RunIgProf TTbar --RunCallgrind TTbar --RunMemcheck TTbar --RunDigiPileUp TTbar --PUInputFile /store/relval/CMSSW_2_2_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/101C84AF-56C4-DD11-A90D-001D09F24EC0.root --cmsdriver="--eventcontent FEVTDEBUGHLT --conditions FrontierConditions_GlobalTag,IDEAL_V9::All"
    (this will run the suite with 5 events for TimeSize tests on MinBias and TTbar, 2 for IgProf tests on TTbar only, 1 for Callgrind tests on TTbar only, 5 for Memcheck on MinBias and TTbar, it will also run DIGI PILEUP for all TTbar tests defined, i.e. 5 TimeSize, 2 IgProf, 1 Callgrind, 5 Memcheck. The file /store/relval/CMSSW_2_2_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/101C84AF-56C4-DD11-A90D-001D09F24EC0.root will be copied locally as INPUT_PILEUP_EVENTS.root and it will be used as the input file for the MixingModule pile up events. All these tests will be done for the step GEN-HLT, i.e. GEN,SIM,DIGI,L1,DIGI2RAW,HLT at once)
    OR
    cmsPerfSuite.py --step GEN-HLT -t 5 -i 2 -c 1 -m 5 --RunTimeSize MinBias,TTbar --RunIgProf TTbar --RunCallgrind TTbar --RunMemcheck TTbar --RunTimeSizePU TTbar --PUInputFile /store/relval/CMSSW_2_2_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/101C84AF-56C4-DD11-A90D-001D09F24EC0.root
    (this will run the suite with 5 events for TimeSize tests on MinBias and TTbar, 2 for IgProf tests on TTbar only, 1 for Callgrind tests on TTbar only, 5 for Memcheck on MinBias and TTbar, it will also run DIGI PILEUP on TTbar but only for 5 TimeSize events. All these tests will be done for the step GEN-HLT, i.e. GEN,SIM,DIGI,L1,DIGI2RAW,HLT at once)
    OR
    cmsPerfSuite.py --step GEN-HLT -t 5 -i 2 -c 1 -m 5 --RunTimeSize MinBias,TTbar --RunIgProf TTbar --RunCallgrind TTbar --RunMemcheck TTbar --RunTimeSizePU TTbar --PUInputFile /store/relval/CMSSW_2_2_1/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/101C84AF-56C4-DD11-A90D-001D09F24EC0.root --cmsdriver="--eventcontent RAWSIM --conditions FrontierConditions_GlobalTag,IDEAL_V9::All"
    (this will run the suite with 5 events for TimeSize tests on MinBias and TTbar, 2 for IgProf tests on TTbar only, 1 for Callgrind tests on TTbar only, 5 for Memcheck on MinBias and TTbar, it will also run DIGI PILEUP on TTbar but only for 5 TimeSize events. All these tests will be done for the step GEN-HLT, i.e. GEN,SIM,DIGI,L1,DIGI2RAW,HLT at once. It will also add the options "--eventcontent RAWSIM --conditions FrontierConditions_GlobalTag,IDEAL_V9::All" to all cmsDriver.py commands executed by the suite. In addition it will run only 2 cmsDriver.py "steps": "GEN,SIM" and "DIGI". Note the syntax GEN-SIM for combined cmsDriver.py steps)
    
    Legal entries for individual candles (--RunTimeSize, --RunIgProf, --RunCallgrind, --RunMemcheck options):
    %s
    ''' % ("\n".join(Candles)))
    
        parser.set_defaults(TimeSizeEvents   = 0        ,
                            IgProfEvents     = 0          ,
                            CallgrindEvents  = 0          ,
                            MemcheckEvents   = 0          ,
                            cmsScimark       = 10         ,
                            cmsScimarkLarge  = 10         ,  
                            cmsdriverOptions = "--eventcontent FEVTDEBUGHLT", # Decided to avoid using the automatic parsing of cmsDriver_highstats_hlt.txt: cmsRelValCmd.get_cmsDriverOptions(), #Get these options automatically now!
                            #"Release Integrators" will create another file relative to the performance suite and the operators will fetch from that file the --cmsdriver option... for now just set the eventcontent since that is needed in order for things to run at all now...
                            stepOptions      = ""         ,
                            profilers        = ""         ,
                            outputdir        = ""         ,
                            logfile          = os.path.join(os.getcwd(),"cmsPerfSuite.log"),
                            runonspare       = True       ,
                            bypasshlt        = False      ,
                            quicktest        = False      ,
                            unittest         = False      ,
                            noexec           = False      ,
                            dryrun           = False      ,
                            verbose          = True       ,
                            create           = False      ,
                            previousrel      = ""         ,
                            castordir        = self._CASTOR_DIR,
                            cores            = cmsCpuInfo.get_NumOfCores(), #Get Number of cpu cores on the machine from /proc/cpuinfo
                            cpu              = "1"        , #Cpu core on which the suite is run:
                            RunTimeSize      = ""         ,
                            RunIgProf        = ""         ,
                            RunCallgrind     = ""         ,
                            RunMemcheck      = ""         ,
                            RunDigiPileUP    = ""         ,
                            RunTimeSizePU    = ""         ,
                            RunIgProfPU      = ""         ,
                            RunCallgrindPU   = ""         ,
                            RunMemcheckPU    = ""         ,
                            PUInputFile      = ""         ,
                            userInputFile    = ""         )
        parser.add_option('--createIgVol', action="store_true", dest='create',
            help = 'Create IgProf AFS volume for the release and architecture')
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
        parser.add_option('-c', '--callgrind'  , type='int'   , dest='CallgrindEvents'  , metavar='<#EVENTS>'   ,
            help = 'specify the number of events for the Callgrind tests'                   )
        parser.add_option('-m', '--memcheck'  , type='int'   , dest='MemcheckEvents'  , metavar='<#EVENTS>'   ,
            help = 'specify the number of events for the Memcheck tests'                   )
        parser.add_option('--cmsScimark'      , type='int'   , dest='cmsScimark'      , metavar=''            ,
            help = 'specify the number of times the cmsScimark benchmark is run before and after the performance suite on cpu1')
        parser.add_option('--cmsScimarkLarge' , type='int'   , dest='cmsScimarkLarge' , metavar=''            ,
            help = 'specify the number of times the cmsScimarkLarge benchmark is run before and after the performance suite on cpu1')
        parser.add_option('--cores'           , type='int', dest='cores'              , metavar='<CORES>'     ,
            help = 'specify the number of cores of the machine (can be used with 0 to stop cmsScimark from running on the other cores)')        
        parser.add_option('--cmsdriver' , type='string', dest='cmsdriverOptions', metavar='<OPTION_STR>',
            help = 'specify special options to use with the cmsDriver.py commands (designed for integration build use')        
        parser.add_option('-a', '--archive'   , type='string', dest='castordir'       , metavar='<DIR>'       ,
            help = 'specify the wanted CASTOR directory where to store the results tarball')
        parser.add_option('-L', '--logfile'   , type='string', dest='logfile'         , metavar='<FILE>'      ,
            help = 'file to store log output of the script')                
        parser.add_option('-o', '--output'    , type='string', dest='outputdir'       , metavar='<DIR>'       ,
            help = 'specify the directory where to store the output of the script')        
        parser.add_option('-r', '--prevrel'   , type='string', dest='previousrel'     , metavar='<DIR>'       ,
            help = 'Top level dir of previous release for regression analysis')        
        parser.add_option('--step'            , type='string', dest='stepOptions'     , metavar='<STEPS>'     ,
            help = 'specify the processing steps intended (instead of the default ones)' )
        parser.add_option('--cpu'             , type='string', dest='cpu'             , metavar='<CPU>'       ,
            help = 'specify the core on which to run the performance suite')

        #Adding new options to put everything configurable at command line:
        parser.add_option('--RunTimeSize'             , type='string', dest='RunTimeSize' , metavar='<CANDLES>'       ,
            help = 'specify on which candles to run the TimeSize tests')
        parser.add_option('--RunIgProf'             , type='string', dest='RunIgProf' , metavar='<CANDLES>'       ,
            help = 'specify on which candles to run the IgProf tests')
        parser.add_option('--RunCallgrind'             , type='string', dest='RunCallgrind' , metavar='<CANDLES>'       ,
            help = 'specify on which candles to run the Callgrind tests')
        parser.add_option('--RunMemcheck'             , type='string', dest='RunMemcheck' , metavar='<CANDLES>'       ,
            help = 'specify on which candles to run the Memcheck tests')
        parser.add_option('--RunDigiPileUp'             , type='string', dest='RunDigiPileUp' , metavar='<CANDLES>'       ,
            help = 'specify the candle on which to run DIGI PILE UP and repeat all the tests set to run on that candle with PILE UP')
        parser.add_option('--PUInputFile'             , type='string', dest='PUInputFile' , metavar='<FILE>'       ,
            help = 'specify the root file to pick the pile-up events from')
        parser.add_option('--RunTimeSizePU'             , type='string', dest='RunTimeSizePU' , metavar='<CANDLES>'       ,
            help = 'specify on which candles to run the TimeSize tests with PILE UP')
        parser.add_option('--RunIgProfPU'             , type='string', dest='RunIgProfPU' , metavar='<CANDLES>'       ,
            help = 'specify on which candles to run the IgProf tests with PILE UP')
        parser.add_option('--RunCallgrindPU'             , type='string', dest='RunCallgrindPU' , metavar='<CANDLES>'       ,
            help = 'specify on which candles to run the Callgrind tests with PILE UP')
        parser.add_option('--RunMemcheckPU'             , type='string', dest='RunMemcheckPU' , metavar='<CANDLES>'       ,
            help = 'specify on which candles to run the Memcheck tests with PILE UP')

        #Adding a filein option to use pre-processed RAW file for RECO and HLT:
        parser.add_option('--filein'             , type='string', dest='userInputFile' , metavar='<FILE>', #default="",
            help = 'specify input RAW root file for HLT and RAW2DIGI-RECO (list the files in the same order as the candles for the tests)')

        #Adding an option to handle additional (to the default user) email addresses to the email notification list (that sends the cmsPerfSuite.log once the performance suite is done running):
        parser.add_option('--mail', type='string', dest='MailLogRecipients', metavar='<EMAIL ADDRESS>', default=self.user, help='specify valid email address(es) name@domain in order to receive notification at the end of the performance suite running with the cmsPerfSuite.log file')

        #Adding option to turn off tarball creation at the end of the execution of the performance suite:
        parser.add_option('--no_tarball', action="store_false", dest='tarball', default=True, help='Turn off automatic tarball creation at the end of the performance suite execution')
                
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
        devel.add_option('--no_exec'           , action="store_true", dest='noexec' ,
            help = 'Run the suite without executing the cmsRelvalreport.py commands in the various directories. This is a useful debugging tool.'         )
        parser.add_option_group(devel)
        (options, args) = parser.parse_args(argslist)
    
    
        self._debug           = options.debug
        self._unittest        = options.unittest
        self._noexec          = options.noexec
        self._verbose         = options.verbose
        self._dryrun          = options.dryrun
        create           = options.create
        castordir        = options.castordir
        TimeSizeEvents   = options.TimeSizeEvents
        IgProfEvents     = options.IgProfEvents
        CallgrindEvents  = options.CallgrindEvents
        MemcheckEvents   = options.MemcheckEvents
        cmsScimark       = options.cmsScimark
        cmsScimarkLarge  = options.cmsScimarkLarge
        cmsdriverOptions = options.cmsdriverOptions
        stepOptions      = options.stepOptions
        quicktest        = options.quicktest
        #candleoption     = options.candleOptions
        runonspare       = options.runonspare
        profilers        = options.profilers.strip()
        cpu              = options.cpu.strip()
        bypasshlt        = options.bypasshlt
        cores            = options.cores
        logfile          = options.logfile
        prevrel          = options.previousrel
        outputdir        = options.outputdir
        RunTimeSize      = options.RunTimeSize
        RunIgProf        = options.RunIgProf
        RunCallgrind     = options.RunCallgrind
        RunMemcheck      = options.RunMemcheck
        RunDigiPileUp    = options.RunDigiPileUp
        RunTimeSizePU    = options.RunTimeSizePU
        RunIgProfPU      = options.RunIgProfPU
        RunCallgrindPU   = options.RunCallgrindPU
        RunMemcheckPU    = options.RunMemcheckPU
        PUInputFile      = options.PUInputFile
        userInputFile    = options.userInputFile
        if options.MailLogRecipients !="" and self.user not in options.MailLogRecipients: #To allow for the --mail "" case of suppressing the email and the default user case
           MailLogRecipients= self.user+","+options.MailLogRecipients #Add the user by default if there is a mail report
        else:
           MailLogRecipients=options.MailLogRecipients
        tarball          = options.tarball
    
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
            self.logh.write("WARNING: Please use GEN-SIM with a hypen not a \",\"!\n")
        #Using the step option as a switch between different dictionaries for:
        #RunTimeSize,RunIgProf,RunCallgrind,RunMemCheck,RunDigiPileUp:
        if stepOptions == "" or stepOptions == 'Default':
            pass
        else:
            stepOptions='--usersteps=%s' % (stepOptions)        
    
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
                self.logh.write("ERROR: Previous release dir %s could not be found" % prevrel)
                sys.exit()
    
        #############
        # Setup quicktest option
        #
        if quicktest:
            TimeSizeEvents = 1
            IgProfEvents = 1
            CallgrindEvents = 0
            MemcheckEvents = 0
            cmsScimark = 1
            cmsScimarkLarge = 1
    
        #############
        # Setup unit test option
        #
        if self._unittest:
            self._verbose = False
            if stepOptions == "":
                stepOptions = "GEN-SIM,DIGI,L1,DIGI2RAW,HLT,RAW2DIGI-RECO"
            cmsScimark      = 0
            cmsScimarkLarge = 0
            CallgrindEvents  = 0
            MemcheckEvents  = 0
            IgProfEvents    = 0
            TimeSizeEvents  = 1
        
        #Split all the RunTimeSize etc candles in lists:
        TimeSizeCandles=[]
        IgProfCandles=[]
        CallgrindCandles=[]
        MemcheckCandles=[]
        TimeSizePUCandles=[]
        IgProfPUCandles=[]
        CallgrindPUCandles=[]
        MemcheckPUCandles=[]
        userInputRootFiles=[]
        if RunTimeSize:
            TimeSizeCandles = RunTimeSize.split(",")
        if RunIgProf:
            IgProfCandles = RunIgProf.split(",")
        if RunCallgrind:
            CallgrindCandles = RunCallgrind.split(",")
        if RunMemcheck:
            MemcheckCandles = RunMemcheck.split(",")
        if RunDigiPileUp:
            for candle in RunDigiPileUp.split(","):
                if candle in TimeSizeCandles:
                    TimeSizePUCandles.append(candle)
                if candle in IgProfCandles:
                    IgProfPUCandles.append(candle)
                if candle in CallgrindCandles:
                    CallgrindPUCandles.append(candle)
                if candle in MemcheckCandles:
                    MemcheckPUCandles.append(candle)
        if RunTimeSizePU:
            TimeSizePUCandles.extend(RunTimeSizePU.split(","))
            #Some smart removal of duplicates from the list!
            temp=set(TimeSizePUCandles)
            TimeSizePUCandles=list(temp) #Doing it in 2 steps to avoid potential issues with type of arguments
        if RunIgProfPU:
            IgProfPUCandles.extend(RunIgProfPU.split(","))
            #Some smart removal of duplicates from the list!
            temp=set(IgProfPUCandles)
            IgProfPUCandles=list(temp) #Doing it in 2 steps to avoid potential issues with type of arguments
        if RunCallgrindPU:
            CallgrindPUCandles.extend(RunCallgrindPU.split(","))
            #Some smart removal of duplicates from the list!
            temp=set(CallgrindPUCandles)
            CallgrindPUCandles=list(temp) #Doing it in 2 steps to avoid potential issues with type of arguments
        if RunMemcheckPU:
            MemcheckPUCandles.extend(RunMemcheckPU.split(","))
            #Some smart removal of duplicates from the list!
            temp=set(MemcheckPUCandles)
            MemcheckPUCandles=list(temp) #Doing it in 2 steps to avoid potential issues with type of arguments
        if userInputFile:
           userInputRootFiles=userInputFile.split(",")

           

        #############
        # Setup cmsdriver and eventual cmsdriverPUoption
        #
        cmsdriverPUOptions=""
        if cmsdriverOptions:
            #Set the eventual Pile Up cmsdriver options first:
            if TimeSizePUCandles or IgProfPUCandles or CallgrindPUCandles or MemcheckPUCandles:
                #Bug fixed: no space between --pileup= and LowLumiPileUp (otherwise could omit the =)
                cmsdriverPUOptions = '--cmsdriver="%s %s%s"'%(cmsdriverOptions," --pileup=",cmsDriverPileUpOption)
            #Set the regular ones too:
            cmsdriverOptions = '--cmsdriver="%s"'%cmsdriverOptions        
    
        return (create          ,
                castordir       ,
                TimeSizeEvents  ,
                IgProfEvents    ,
                CallgrindEvents ,
                MemcheckEvents  ,
                cmsScimark      ,
                cmsScimarkLarge ,
                cmsdriverOptions,
                cmsdriverPUOptions,
                stepOptions     ,
                quicktest       ,
                profilers       ,
                cpu             ,
                cores           ,
                prevrel         ,
                bypasshlt       ,
                runonspare      ,
                outputdir       ,
                logfile         ,
                TimeSizeCandles ,
                IgProfCandles   ,
                CallgrindCandles,
                MemcheckCandles ,
                TimeSizePUCandles ,
                IgProfPUCandles   ,
                CallgrindPUCandles,
                MemcheckPUCandles ,
                PUInputFile     ,
                userInputRootFiles,
                MailLogRecipients,
                tarball)
    
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
            self.logh.write(str(command) + "\n")
            self.logh.flush()
    
    #############
    # Run a command and return the exit status
    #
    def runcmd(self,command):
        #Substitute popen with subprocess.Popen!
        #Using try/except until Popen becomes thread safe (it seems that everytime it is called
        #all processes are checked to reap the ones that are done, this creates a race condition with the wait()... that
        #results into an error with "No child process".
        #os.popen(command)
        try:
            process  = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            pid=process.pid
            exitstat= process.wait()
            cmdout   = process.stdout.read()
            exitstat = process.returncode
        except OSError, detail:
            self.logh.write("Race condition in subprocess.Popen has robbed us of the exit code of the %s process (PID %s).Assume it failed!\n %s\n"%(command,pid,detail))
            self.logh.flush()
            exitstat=999
            cmdout="Race condition in subprocess.Popen has robbed us of the exit code of the %s process (PID %s).Assume it failed!\n %s"%(command,pid,detail)
        if self._verbose:
            self.logh.write(cmdout)# + "\n") No need of extra \n!
            self.logh.flush()
        if exitstat == None:
            self.logh.write("Something strange is going on! Exit code was None for command %s: check if it really ran!"%command)
            self.logh.flush()
            exitstat=0
        return exitstat
    
    def getDate(self):
        return time.ctime()
    
    def printDate(self):
        self.logh.write(self.getDate() + "\n")
        self.logh.flush()
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
            #Obsolete popen4-> subprocess.Popen
            #cmdout=os.popen3(cmd)[2].read()
            cmdout=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read()
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
                    self.logh.write("The number of Memcheck event was not changed since the original number of Callgrind event was not 1!\n")
                Outputfile.write(line)
            elif digiRegxp.search(line) and MemcheckRegxp.search(line):
                #Modify
                if NumEvtRegxp.search(line):
                    line=NumEvtRegxp.sub(r"-n 5",line)
                else:
                    self.logh.write("The number of Memcheck event was not changed since the original number of Callgrind event was not 1!\n")
                Outputfile.write(line)
            else:
                Outputfile.write(line)
        self.logh.flush()
        Outputfile.close()
            
        #self.runCmdSet(cmds)
    
    ##################
    # Run cmsScimark benchmarks a number of times
    #
    def benchmarks(self,cpu,pfdir,name,bencher,large=False):
        cmd = self.Commands[cpu][3]
        redirect = ""
        if large:
            redirect = " -large >>"    
        else:
            redirect = " >>"
    
        for i in range(bencher):
           #Check first for the existence of the file so that we can append:
           if not os.path.exists(os.path.join(pfdir,os.path.basename(name))):
              #Equivalent of touch to make sure the file exist to be able to append to it.
              open(os.path.join(pfdir,os.path.basename(name)))
              
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
    def runCmsInput(self,cpu,dir,numevents,candle,cmsdrvopts,stepopt,profiles,bypasshlt,userInputFile):

        #Crappy fix for optional options with special synthax (bypasshlt and userInputFile)
        bypass = ""
        if bypasshlt:
            bypass = "--bypass-hlt"
        userInputFileOption=""
        if userInputFile:
           userInputFileOption = "--filein %s"%userInputFile
        cmd = self.Commands[cpu][2]
        cmds=[]
        #print cmds
        cmds = ("cd %s"                    % (dir),
                "%s %s \"%s\" %s %s %s %s %s" % (cmd,
                                              numevents,
                                              candle,
                                              profiles,
                                              cmsdrvopts,
                                              stepopt,
                                              bypass,userInputFileOption))
        exitstat=0
        exitstat = self.runCmdSet(cmds)
        if self._unittest and (not exitstat == 0):
            self.logh.write("ERROR: CMS Report Input returned a non-zero exit status \n" )
        return exitstat
    ##############
    # Prepares the profiling directory and runs all the selected profiles (if this is not a unit test)
    #
    #Making parameters named to facilitate the handling of arguments (especially with the threading use case)
    def simpleGenReport(self,cpus,perfdir=os.getcwd(),NumEvents=1,candles=['MinBias'],cmsdriverOptions='',stepOptions='',Name='',profilers='',bypasshlt='',userInputRootFiles=''):
        callgrind = Name == "Callgrind"
        memcheck  = Name == "Memcheck"
    
        profCodes = {"TimeSize" : "0123",
                     "IgProf"   : "4567",
                     "IgProf_Perf":"47", #Added the Analyse to IgProf_Perf #FIXME: At the moment Analyse is always run whether 7 is selected or not! Issue to solve in cmsRelvalreportInput.py... but not really important (it's always been there, not impacting our use-cases).
                     "IgProf_Mem":"567",
                     "Callgrind": "8",
                     "Memcheck" : "9",
                     None       : "-1"} 
    
        profiles = profCodes[Name]
        if not profilers == "":
            profiles = profilers        
    
        RelvalreportExitCode=0
        
        for cpu in cpus:
            pfdir = perfdir
            if len(cpus) > 1:
                pfdir = os.path.join(perfdir,"cpu_%s" % cpu)
            for candle in candles:
                #Create the directory for cmsRelvalreport.py running (e.g. MinBias_TimeSize, etc)
                #Catch the case of PILE UP:
                if "--pileup" in cmsdriverOptions:
                   candlename=candle+"_PU"
                else:
                   candlename=candle
                adir=self.mkCandleDir(pfdir,candlename,Name)
                if self._unittest:
                    # Run cmsDriver.py
                    if userInputRootFiles:
                       self.logh.write(userInputRootFiles)
                       userInputFile=userInputRootFiles[0]
                    else:
                       userInputFile=""
                    self.logh.flush()
                    self.runCmsInput(cpu,adir,NumEvents,candle,cmsdriverOptions,stepOptions,profiles,bypasshlt,userInputFile) 
                    self.testCmsDriver(cpu,adir,candle)
                else:
                    if userInputRootFiles:
                       self.logh.write("Variable userInputRootFiles is %s\n"%userInputRootFiles)
                       #Need to use regexp, cannot rely on the order... since for different tests there are different candles...
                       #userInputFile=userInputRootFiles[candles.index(candle)]
                       #FIXME:
                       #Note the issue that the input files HAVE to have in their name the candle as is used in cmsPerfSuite.py command line!
                       #This is currently caught by a printout in the log: should be either taken care of with some exception to throw?
                       #Will put this in the documentation
                       userInputFile=""
                       candleregexp=re.compile(candle)
                       for file in userInputRootFiles:
                          if candleregexp.search(file):
                             userInputFile=file
                             self.logh.write("For these %s %s tests will use user input file %s\n"%(candlename,Name,userInputFile))
                       if userInputFile == "":
                          self.logh.write("***WARNING: For these %s %s tests could not find a matching input file in %s: will try to do without it!!!!!\n"%(candlename,Name,userInputRootFiles))
                       self.logh.flush()
                    else:
                       userInputFile=""
                    DummyTestName=candlename+"_"+stepOptions.split("=")[1]
                    DummyTimer=PerfSuiteTimer(start=datetime.datetime.now()) #Start the timer (DummyTimer is just a reference, but we will use the dictionary to access this later...
                    TimerInfo[Name].update({DummyTestName:DummyTimer}) #Add the TimeSize timer to the dictionary
                    #The following command will create the appropriate SimulationCandlesX.txt file in the relevant directory, ready to run cmsRelvalreport.py on it.
                    self.runCmsInput(cpu,adir,NumEvents,candle,cmsdriverOptions,stepOptions,profiles,bypasshlt,userInputFile)            
                    #Here where the no_exec option kicks in (do everything but do not launch cmsRelvalreport.py, it also prevents cmsScimark spawning...):
                    if self._noexec:
                        self.logh.write("Running in debugging mode, without executing cmsRelvalreport.py\n")
                        self.logh.flush()
                        pass
                    else:
                        #The following command will launch cmsRelvalreport.py on the SimulationCandlesX.txt input file created above.
                        ExitCode=self.runCmsReport(cpu,adir,candle)
                        self.logh.write("Individual cmsRelvalreport.py ExitCode %s\n"%ExitCode)
                        RelvalreportExitCode=RelvalreportExitCode+ExitCode
                        self.logh.write("Summed cmsRelvalreport.py ExitCode %s\n"%RelvalreportExitCode)
                        self.logh.flush()
                    DummyTimer.set_end(datetime.datetime.now())
                    
                    #for proflog in proflogs:
                    #With the change from 2>1&|tee to >& to preserve exit codes, we need now to check all logs...
                    #less nice... we might want to do this externally so that in post-processing its a re-usable tool
                    globpath = os.path.join(adir,"*.log") #"%s.log"%candle)
                    self.logh.write("Looking for logs that match %s\n" % globpath)
                    logs     = glob.glob(globpath)
                    for log in logs:
                        self.logh.write("Found log %s\n" % log)
                        self.displayErrors(log)
        self.printFlush("Returned cumulative RelvalreportExitCode is %s"%RelvalreportExitCode)
        return RelvalreportExitCode
    
    ############
    # Runs benchmarking, cpu spinlocks on spare cores and profiles selected candles
    #
    #FIXME:
    #Could redesign interface of functions to use keyword arguments:
    #def runPerfSuite(**opts):
    #then instead of using castordir variable, would use opts['castordir'] etc    
    def runPerfSuite(self,
                     create           = False,
                     #Swtiching from CASTOR to EOS (using xrdcp instead of rfcp and  root://eoscms//eos/ instead of /castor/cern.ch/
                     #Actually not yet... for consistency we will keep it on CASTOR for now
                     #FIXME! Do the migration, following its implication in PerfDB application!
                     castordir        = "/castor/cern.ch/cms/store/relval/performance/",
                     TimeSizeEvents   = 100        ,
                     IgProfEvents     = 5          ,
                     CallgrindEvents  = 1          ,
                     MemcheckEvents   = 5          ,
                     cmsScimark       = 10         ,
                     cmsScimarkLarge  = 10         ,
                     cmsdriverOptions = ""         ,#Could use directly cmsRelValCmd.get_Options()
                     cmsdriverPUOptions= ""        ,
                     stepOptions      = ""         ,
                     quicktest        = False      ,
                     profilers        = ""         ,
                     cpus             = [1]        ,
                     cpu_list         = [1]        ,
                     cores            = 4          ,#Could use directly cmsCpuInfo.get_NumOfCores()
                     prevrel          = ""         ,
                     bypasshlt        = False      ,
                     runonspare       = True       ,
                     perfsuitedir     = os.getcwd(),
                     logfile          = None,
                     TimeSizeCandles      = ""         ,
                     IgProfCandles        = ""         ,
                     CallgrindCandles     = ""         ,
                     MemcheckCandles      = ""         ,
                     TimeSizePUCandles    = ""         ,
                     IgProfPUCandles      = ""         ,
                     CallgrindPUCandles   = ""         ,
                     MemcheckPUCandles    = ""         ,
                     PUInputFile          = ""         ,
                     userInputFile        = ""         ,
                     MailLogRecipients    = ""         ,
                     tarball              = ""         ):
        
        #Set up a variable for the FinalExitCode to be used as the sum of exit codes:
        FinalExitCode=0

        #Set up the logfile first!
        if not logfile == None:
           try:
              self.logh = open(logfile,"a")
           except (OSError, IOError), detail:
              self.logh.write(detail + "\n")
              self.logh.flush()  

        #Adding HEPSPEC06 score if available in /build/HEPSPEC06.score file
        self.HEPSPEC06 = 0 #Set it to 0 by default (so it is easy to catch in the DB too)
        try:
           HEPSPEC06_file=open("/build/HEPSPEC06.score","r")
           for line in HEPSPEC06_file.readlines():
              if not line.startswith("#") and "HEPSPEC06" in line:
                 self.HEPSPEC06= line.split()[2]
        except IOError:
           self.logh.write("***Warning***: Could not find file /build/HEPSPEC06.score file on this machine!\n")
           self.logh.flush()

        #Adding a copy of /proc/cpuinfo and /proc/meminfo in the working directory so it can be kept in the tarball on CASTOR:
        localcpuinfo=os.path.join(perfsuitedir,"cpuinfo")
        cpuinfo_exitcode=-1
        if os.path.exists(localcpuinfo):
           cpuinfo_exitcode=0
        else:
           self.logh.write("Copying /proc/cpuinfo in current working directory (%s)\n"%perfsuitedir)
           cpuinfo_exitcode=self.runcmd("cp /proc/cpuinfo %s"%perfsuitedir)
        localmeminfo=os.path.join(perfsuitedir,"meminfo")
        meminfo_exitcode=-1
        if os.path.exists(localmeminfo):
           meminfo_exitcode=0
        else:
           self.logh.write("Copying /proc/meminfo in current working directory (%s)\n"%perfsuitedir)
           meminfo_exitcode=self.runcmd("cp /proc/meminfo %s"%perfsuitedir)
        if cpuinfo_exitcode or meminfo_exitcode:
           self.logh.write("There was an issue copying the cpuinfo or meminfo files!\n")
        self.logh.flush()
        
        try:        
            if not prevrel == "":
                self.logh.write("Production of regression information has been requested with release directory %s\n" % prevrel)
            if not cmsdriverOptions == "":
                self.logh.write("Running cmsDriver.py with user defined options: %s\n" % cmsdriverOptions)
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
            self.logh.write("Current Architecture is %s\n"%self.cmssw_arch)
            self.logh.write("Current CMSSW version is %s\n"%self.cmssw_version)
            self.logh.write("This machine ( %s ) is assumed to have %s cores, and the suite will be run on cpu %s\n" %(self.host,cores,cpus))
            self.logh.write("This machine's HEPSPEC06 score is: %s \n"%self.HEPSPEC06)
            path=os.path.abspath(".")
            self.logh.write("Performance Suite started running at %s on %s in directory %s, run by user %s\n" % (self.getDate(),self.host,path,self.user))
            #Start the timer for the total performance suite running time:
            TotalTime=PerfSuiteTimer(start=datetime.datetime.now())
            #Also initialize the dictionary that will contain all the timing information:
            global TimerInfo
            TimerInfo={'TotalTime':{'TotalTime':TotalTime}} #Structure will be {'key':[PerfSuiteTimerInstance,...],...}
            #Obsolete popen4-> subprocess.Popen
            #showtags=os.popen4("showtags -r")[1].read()
            showtags=subprocess.Popen("showtags -r",shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read()
            self.logh.write(showtags) # + "\n") No need for extra \n!
            self.logh.flush()
            #For the log:
            if self._verbose:
                self.logh.write("The performance suite results tarball will be stored in CASTOR at %s\n" % self._CASTOR_DIR)
                self.logh.write("%s TimeSize events\n" % TimeSizeEvents)
                self.logh.write("%s IgProf events\n"   % IgProfEvents)
                self.logh.write("%s Callgrind events\n" % CallgrindEvents)
                self.logh.write("%s Memcheck events\n" % MemcheckEvents)
                self.logh.write("%s cmsScimark benchmarks before starting the tests\n"      % cmsScimark)
                self.logh.write("%s cmsScimarkLarge benchmarks before starting the tests\n" % cmsScimarkLarge)
                self.logh.flush()
            #Actual script actions!
            #Will have to fix the issue with the matplotlib pie-charts:
            #Used to source /afs/cern.ch/user/d/dpiparo/w0/perfreport2.1installation/share/perfreport/init_matplotlib.sh
            #Need an alternative in the release

            #Code for the architecture benchmarking use-case
            if len(cpus) > 1:
                for cpu in cpus:
                    cpupath = os.path.join(perfsuitedir,"cpu_%s" % cpu)
                    if not os.path.exists(cpupath):
                        os.mkdir(cpupath)
            
            self.Commands = {}
            AllScripts = self.Scripts + self.AuxiliaryScripts
    
            for cpu in range(cmsCpuInfo.get_NumOfCores()): #FIXME use the actual number of cores of the machine here!
                self.Commands[cpu] = []

            #Information for the log:
            self.logh.write("Full path of all the scripts used in this run of the Performance Suite:\n")
            for script in AllScripts:
                which="which " + script
    
                #Logging the actual version of cmsDriver.py, cmsRelvalreport.py, cmsSimPyRelVal.pl
                #Obsolete popen4-> subprocess.Popen
                #whichstdout=os.popen4(which)[1].read()
                whichstdout=subprocess.Popen(which,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read()
                self.logh.write(whichstdout) # + "\n") No need of the extra \n!
                if script in self.Scripts:
                    for cpu in range(cmsCpuInfo.get_NumOfCores()):#FIXME use the actual number of cores of the machine here!
                        command="taskset -c %s %s" % (cpu,script)
                        self.Commands[cpu].append(command)
                        
            #First submit the cmsScimark benchmarks on the unused cores:
            scimark = ""
            scimarklarge = ""
            if not (self._unittest or self._noexec):
                if (len(cpu_list) != cores):
                    for core in range(cores):
                        if (not core in cpus) and runonspare:
                            self.logh.write("Submitting cmsScimarkLaunch.csh to run on core cpu "+str(core) + "\n")
                            subcmd = "cd %s ; cmsScimarkLaunch.csh %s" % (perfsuitedir, str(core))            
                            command="taskset -c %s sh -c \"%s\" &" % (str(core), subcmd)
                            self.logh.write(command + "\n")
                      
                            #cmsScimarkLaunch.csh is an infinite loop to spawn cmsScimark2 on the other
                            #cpus so it makes no sense to try reading its stdout/err
                            #Obsolete popen4-> subprocess.Popen
                            #os.popen4(command)
                            subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                      
            self.logh.flush()
    
            #Don't do benchmarking if in debug mode... saves time
            benching = not self._debug
            ##FIXME:
            #We may want to introduce a switch here or agree on a different default (currently 10 cmsScimark and 10 cmsScimarkLarge)
            if benching and not (self._unittest or self._noexec): 
                #Submit the cmsScimark benchmarks on the cpu where the suite will be run:
                for cpu in cpus:
                    scimark      = open(os.path.join(perfsuitedir,"cmsScimark2.log")      ,"w")        
                    scimarklarge = open(os.path.join(perfsuitedir,"cmsScimark2_large.log"),"w")
                    if cmsScimark > 0:
                        self.logh.write("Starting with %s cmsScimark on cpu%s\n"       % (cmsScimark,cpu))
                        cmsScimarkInitialTime=PerfSuiteTimer(start=datetime.datetime.now()) #Create the cmsScimark PerfSuiteTimer
                        TimerInfo.update({'cmsScimarkTime':{'cmsScimarkInitial':cmsScimarkInitialTime}}) #Add the cmsScimarkInitialTime information to the general TimerInfo dictionary
                        self.benchmarks(cpu,perfsuitedir,scimark.name,cmsScimark)
                        cmsScimarkInitialTime.set_end(datetime.datetime.now()) #Stop the cmsScimark initial timer
    
                    if cmsScimarkLarge > 0:
                        self.logh.write("Following with %s cmsScimarkLarge on cpu%s\n" % (cmsScimarkLarge,cpu))
                        cmsScimarkLargeInitialTime=PerfSuiteTimer(start=datetime.datetime.now()) #Create the cmsScimarkLarge PerfSuiteTimer
                        TimerInfo['cmsScimarkTime'].update({'cmsScimarkLargeInitial':cmsScimarkLargeInitialTime}) #Add the cmsScimarkLargeInitialTime information to the general TimerInfo dictionary
                        self.benchmarks(cpu,perfsuitedir,scimarklarge.name,cmsScimarkLarge, large=True)
                        cmsScimarkLargeInitialTime.set_end(datetime.datetime.now()) #Stop the cmsScimarkLarge Initial timer
                self.logh.flush()
            #Handling the Pile up input file here:
            if (TimeSizePUCandles or IgProfPUCandles or CallgrindPUCandles or MemcheckPUCandles) and not ("FASTSIM" in stepOptions):
                #Note the FASTSIM exclusion... since there is no need to copy the file for FASTSIM.
                PUInputName=os.path.join(perfsuitedir,"INPUT_PILEUP_EVENTS.root")
                if PUInputFile:
                    #Define the actual command to copy the file locally:
                    #Allow the file to be mounted locally (or accessible via AFS)
                    copycmd="cp"
                    #Allow the file to be on CASTOR (taking a full CASTOR path)
                    if '/store/relval/' in PUInputFile:
                       #Switching from CASTOR to EOS, i.e. from rfcp to xrdcp
                        copycmd="xrdcp"
                        #Accept plain LFNs from DBS for RelVal CASTOR files:
                        #Minor fix to allow the case of user using the full path /castor/cern.ch/cms...
                        if PUInputFile.startswith('/store/relval/'):
                           #Switching to EOS from CASTOR:
                           #PUInputFile="/castor/cern.ch/cms"+PUInputFile
                           PUInputFile="root://eoscms//eos/cms"+PUInputFile
                    #Copy the file locally
                    self.logh.write("Copying the file %s locally to %s\n"%(PUInputFile,PUInputName))
                    self.logh.flush()
                    GetPUInput=subprocess.Popen("%s %s %s"%(copycmd,PUInputFile,PUInputName), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    GetPUInputExitCode=GetPUInput.wait()
                    #Allow even the potential copy of a local file (even one already named INPUT_PILEUP_EVENTS.root!)
                    if GetPUInputExitCode:
                        self.logh.write("The copying of the pile-up input file returned a non-zero exit code: %s \nThis is the stdout+stderr if the command:\n%s\n"%(GetPUInputExitCode,GetPUInput.stdout))
                #Ultimately accept the case of the file being already there and not being specified in the --PUInputFile option
                if not os.path.exists(PUInputName):
                    self.logh.write("The necessary INPUT_PILEUP_EVENTS.root file was not found in the working directory %s\nExiting now!"%perfsuitedir)
                    self.logh.flush()
                    sys.exit(1)
                else:
                    #Set up here the DIGI PILE UP options
                    self.printFlush("Some PILE UP tests will be run!")
                    #Actually setting them earlier... when handling options... May not need this else after all... or just as a log entry.
                    self.printFlush("cmsdriverPUOptions is %s"%cmsdriverPUOptions)
                    pass
            
            #TimeSize tests:
            if TimeSizeEvents > 0:
               TimeSizeTime=PerfSuiteTimer(start=datetime.datetime.now()) #Start the TimeSize timer
               TimerInfo.update({'TimeSize':{'TotalTime':TimeSizeTime}}) #Add the TimeSize timer to the dictionary
               if TimeSizeCandles:
                  self.logh.write("Launching the TimeSize tests (TimingReport, TimeReport, SimpleMemoryCheck, EdmSize) with %s events each\n" % TimeSizeEvents)
                  NoPileUpTime=PerfSuiteTimer(start=datetime.datetime.now()) #Start the TimeSize timer
                  TimerInfo['TimeSize'].update({'NoPileUpTime':NoPileUpTime}) #Add the TimeSize No Pile Up tests timer to the list
                  self.printDate()
                  self.logh.flush()
                  ReportExit=self.simpleGenReport(cpus,perfsuitedir,TimeSizeEvents,TimeSizeCandles,cmsdriverOptions,stepOptions,"TimeSize",profilers,bypasshlt,userInputFile)
                  FinalExitCode=FinalExitCode+ReportExit
                  #Adding a time stamp here to parse for performance suite running time data
                  self.printFlush("Regular TimeSize tests were finished at %s"%(self.getDate()))
                  NoPileUpTime.set_end(datetime.datetime.now()) #Stop TimeSize timer
               
               #Launch eventual Digi Pile Up TimeSize too:
               if TimeSizePUCandles:
                  self.logh.write("Launching the PILE UP TimeSize tests (TimingReport, TimeReport, SimpleMemoryCheck, EdmSize) with %s events each\n" % TimeSizeEvents)
                  PileUpTime=PerfSuiteTimer(start=datetime.datetime.now()) #Start the TimeSize timer
                  TimerInfo['TimeSize'].update({'PileUpTime':PileUpTime}) #Add the TimeSize Pile Up tests timer to the list
                  self.printDate()
                  self.logh.flush()
                  ReportExit=self.simpleGenReport(cpus,perfsuitedir,TimeSizeEvents,TimeSizePUCandles,cmsdriverPUOptions,stepOptions,"TimeSize",profilers,bypasshlt,userInputFile)
                  FinalExitCode=FinalExitCode+ReportExit
                  #Adding a time stamp here to parse for performance suite running time data
                  self.printFlush("Pileup TimeSize tests were finished at %s"%(self.getDate()))
                  PileUpTime.set_end(datetime.datetime.now()) #Stop TimeSize timer
                  
               #Check for issue with 
               if not (TimeSizeCandles or TimeSizePUCandles):
                  self.printFlush("A number of events (%s) for TimeSize tests was selected, but no candle for regular or pileup tests was selected!"%(TimeSizeEvents))
               #Adding a time stamp here to parse for performance suite running time data
               self.printFlush("All TimeSize tests were finished at %s"%(self.getDate()))
               TimeSizeTime.set_end(datetime.datetime.now()) #Stop TimeSize timer
            
            #Stopping all cmsScimark jobs and analysing automatically the logfiles
            #No need to waste CPU while the load does not affect Valgrind measurements!
            if not (self._unittest or self._noexec):
                self.logh.write("Stopping all cmsScimark jobs now\n")
                subcmd = "cd %s ; %s" % (perfsuitedir,self.AuxiliaryScripts[2])
                stopcmd = "sh -c \"%s\"" % subcmd
                self.printFlush(stopcmd)
                #os.popen(stopcmd)
                #Obsolete popen4-> subprocess.Popen
                #self.printFlush(os.popen4(stopcmd)[1].read())
                self.printFlush(subprocess.Popen(stopcmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.read())

            #From here on we can use all available cores to speed up the performance suite remaining tests:
            if cores==0: #When specifying the cpu to run the suite on, one has to set cores to 0 to avoid threading of PerfSuite itself...
                                          #So we need to catch this case for the IB tests case where we assign the test to a specific cpu.
                AvailableCores=cpus
            elif len(cpu_list) == cores: # For the new relval case, when running all the tests on one machine,
                                         # specifying the same number of cores and cpus (like: --cores 3, --cpu 3,4,5)
                AvailableCores=cpus
            else:
                AvailableCores=range(cores)
                
            #Initialize a list that will contain all the simpleGenReport keyword arguments (1 dictionary per test):
            TestsToDo=[]
            #IgProf tests:
            if IgProfEvents > 0:
               if IgProfCandles:
                   self.printFlush("Preparing IgProf tests")
                   #Special case for IgProf: user could pick with the option --profilers to run only IgProf perf or Mem (or Mem_Total alone etc)
                   #So in general we want to be able to split the perf and mem tests...
                   #For the case of --profiler option we will run only 1 test (i.e. it will get one core slot until it is done with whatever profiling choosen)
                   if profilers:
                      self.printFlush("Special profiler option for IgProf was indicated by the user: %s"%profilers)
                      #Prepare the simpleGenReport arguments for this test:
                      IgProfProfilerArgs={
                         'perfdir':perfsuitedir,
                         'NumEvents':IgProfEvents,
                         'candles':IgProfCandles,
                         'cmsdriverOptions':cmsdriverOptions,
                         'stepOptions':stepOptions,
                         'Name':"IgProf",
                         'profilers':profilers,
                         'bypasshlt':bypasshlt,
                         'userInputRootFiles':userInputFile
                         }
                      #Append the test to the TestsToDo list:
                      TestsToDo.append(IgProfProfilerArgs)
                      self.printFlush("Appended IgProf test with profiler option %s to the TestsToDo list"%profilers)
                   #For the default case (4,5,6,7) we split the tests into 2 jobs since they naturally are 2 cmsRun jobs and for machines with many cores this will
                   #make the performance suite run faster.
                   else:
                      self.printFlush("Splitting the IgProf tests into Perf and Mem to parallelize the cmsRun execution as much as possible:")
                      ##PERF##
                      #Prepare the simpleGenReport arguments for this test:
                      IgProfPerfArgs={
                         'perfdir':perfsuitedir,
                         'NumEvents':IgProfEvents,
                         'candles':IgProfCandles,
                         'cmsdriverOptions':cmsdriverOptions,
                         'stepOptions':stepOptions,
                         'Name':"IgProf_Perf",
                         'profilers':profilers,
                         'bypasshlt':bypasshlt,
                         'userInputRootFiles':userInputFile
                         }
                      #Append the test to the TestsToDo list:
                      TestsToDo.append(IgProfPerfArgs)
                      self.printFlush("Appended IgProf PERF test to the TestsToDo list")
                      ##MEM##
                      #Prepare the simpleGenReport arguments for this test:
                      IgProfMemArgs={
                         'perfdir':perfsuitedir,
                         'NumEvents':IgProfEvents,
                         'candles':IgProfCandles,
                         'cmsdriverOptions':cmsdriverOptions,
                         'stepOptions':stepOptions,
                         'Name':"IgProf_Mem",
                         'profilers':profilers,
                         'bypasshlt':bypasshlt,
                         'userInputRootFiles':userInputFile
                         }
                      #Append the test to the TestsToDo list:
                      TestsToDo.append(IgProfMemArgs)
                      self.printFlush("Appended IgProf MEM test to the TestsToDo list")
                #The following will be handled in the while loop that handles the starting of the threads:
                #ReportExit=self.simpleGenReport(cpus,perfsuitedir,IgProfEvents,IgProfCandles,cmsdriverOptions,stepOptions,"IgProf",profilers,bypasshlt,userInputFile)
                #FinalExitCode=FinalExitCode+ReportExit
                #Launch eventual Digi Pile Up IgProf too:
               if IgProfPUCandles:
                   self.printFlush("Preparing IgProf PileUp tests")
                   #Special case for IgProf: user could pick with the option --profilers to run only IgProf perf or Mem (or Mem_Total alone etc)
                   #So in general we want to be able to split the perf and mem tests...
                   #For the case of --profiler option we will run only 1 test (i.e. it will get one core slot until it is done with whatever profiling choosen)
                   if profilers:
                      self.printFlush("Special profiler option for IgProf was indicated by the user: %s"%profilers)
                      #Prepare the simpleGenReport arguments for this test:
                      IgProfProfilerPUArgs={
                         'perfdir':perfsuitedir,
                         'NumEvents':IgProfEvents,
                         'candles':IgProfPUCandles,
                         'cmsdriverOptions':cmsdriverPUOptions,
                         'stepOptions':stepOptions,
                         'Name':"IgProf",
                         'profilers':profilers,
                         'bypasshlt':bypasshlt,
                         'userInputRootFiles':userInputFile
                         }
                      #Append the test to the TestsToDo list:
                      TestsToDo.append(IgProfProfilerPUArgs)
                      self.printFlush("Appended IgProf PileUp test with profiler option %s to the TestsToDo list"%profilers)
                   else:
                      self.printFlush("Splitting the IgProf tests into Perf and Mem to parallelize the cmsRun execution as much as possible:")
                      ##PERF##
                      #Prepare the simpleGenReport arguments for this test:
                      IgProfPerfPUArgs={
                         'perfdir':perfsuitedir,
                         'NumEvents':IgProfEvents,
                         'candles':IgProfPUCandles,
                         'cmsdriverOptions':cmsdriverPUOptions,
                         'stepOptions':stepOptions,
                         'Name':"IgProf_Perf",
                         'profilers':profilers,
                         'bypasshlt':bypasshlt,
                         'userInputRootFiles':userInputFile
                         }
                      #Append the test to the TestsToDo list:
                      TestsToDo.append(IgProfPerfPUArgs)
                      self.printFlush("Appended IgProf MEM PileUp test to the TestsToDo list")
                      ##MEM##
                      #Prepare the simpleGenReport arguments for this test:
                      IgProfMemPUArgs={
                         'perfdir':perfsuitedir,
                         'NumEvents':IgProfEvents,
                         'candles':IgProfPUCandles,
                         'cmsdriverOptions':cmsdriverPUOptions,
                         'stepOptions':stepOptions,
                         'Name':"IgProf_Mem",
                         'profilers':profilers,
                         'bypasshlt':bypasshlt,
                         'userInputRootFiles':userInputFile
                         }
                      #Append the test to the TestsToDo list:
                      TestsToDo.append(IgProfMemPUArgs)
                      self.printFlush("Appended IgProf MEM PileUp test to the TestsToDo list")
               if not (IgProfCandles or IgProfPUCandles):
                   self.printFlush("A number of events (%s) for IgProf tests was selected, but no candle for regular or pileup tests was selected!"%(IgProfEvents))
               
                    
            #Valgrind tests:
            if CallgrindEvents > 0:
               if CallgrindCandles:
                  self.printFlush("Preparing Callgrind tests")
                  CallgrindArgs={
                     'perfdir':perfsuitedir,
                     'NumEvents':CallgrindEvents,
                     'candles':CallgrindCandles,
                     'cmsdriverOptions':cmsdriverOptions,
                     'stepOptions':stepOptions,
                     'Name':"Callgrind",
                     'profilers':profilers,
                     'bypasshlt':bypasshlt,
                     'userInputRootFiles':userInputFile
                     }
                  #Append the test to the TestsToDo list:
                  TestsToDo.append(CallgrindArgs)
                  self.printFlush("Appended Callgrind test to the TestsToDo list")
               #Launch eventual Digi Pile Up Callgrind too:
               if CallgrindPUCandles:
                  self.printFlush("Preparing Callgrind PileUp tests")
                  CallgrindPUArgs={
                     'perfdir':perfsuitedir,
                     'NumEvents':CallgrindEvents,
                     'candles':CallgrindPUCandles,
                     'cmsdriverOptions':cmsdriverPUOptions,
                     'stepOptions':stepOptions,
                     'Name':"Callgrind",
                     'profilers':profilers,
                     'bypasshlt':bypasshlt,
                     'userInputRootFiles':userInputFile
                     }
                  #Append the test to the TestsToDo list:
                  TestsToDo.append(CallgrindPUArgs)
                  self.printFlush("Appended Callgrind PileUp test to the TestsToDo list")
               if not (CallgrindCandles or CallgrindPUCandles):
                  self.printFlush("A number of events (%s) for Callgrind tests was selected, but no candle for regular or pileup tests was selected!"%(CallgrindEvents))
                  
            if MemcheckEvents > 0:
               if MemcheckCandles:
                  self.printFlush("Preparing Memcheck tests")
                  MemcheckArgs={
                     'perfdir':perfsuitedir,
                     'NumEvents':MemcheckEvents,
                     'candles':MemcheckCandles,
                     'cmsdriverOptions':cmsdriverOptions,
                     'stepOptions':stepOptions,
                     'Name':"Memcheck",
                     'profilers':profilers,
                     'bypasshlt':bypasshlt,
                     'userInputRootFiles':userInputFile
                     }
                  #Append the test to the TestsToDo list:
                  TestsToDo.append(MemcheckArgs)
                  self.printFlush("Appended Memcheck test to the TestsToDo list")
               #Launch eventual Digi Pile Up Memcheck too:
               if MemcheckPUCandles:
                  self.printFlush("Preparing Memcheck PileUp tests")
                  MemcheckPUArgs={
                     'perfdir':perfsuitedir,
                     'NumEvents':MemcheckEvents,
                     'candles':MemcheckPUCandles,
                     'cmsdriverOptions':cmsdriverPUOptions,
                     'stepOptions':stepOptions,
                     'Name':"Memcheck",
                     'profilers':profilers,
                     'bypasshlt':bypasshlt,
                     'userInputRootFiles':userInputFile
                     }
                  #Append the test to the TestsToDo list:
                  TestsToDo.append(MemcheckPUArgs)  
                  self.printFlush("Appended Memcheck PileUp test to the TestsToDo list")
               if not (MemcheckCandles or MemcheckPUCandles):
                  self.printFlush("A number of events (%s) for Memcheck tests was selected, but no candle for regular or pileup tests was selected!"%(MemcheckEvents))
                  
            #Here if there are any IgProf, Callgrind or MemcheckEvents to be run,
            #run the infinite loop that submits the PerfTest() threads on the available cores:
            if IgProfEvents or CallgrindEvents or MemcheckEvents:
               #FIXME:We should consider what behavior makes most sense in case we use the --cores option at this time only the cores=0 care is considered...
               self.printFlush("Threading all remaining tests on all %s available cores!"%len(AvailableCores))
               self.printDate()
               self.logh.flush()
               #Save the original AvailableCores list to use it as a test to break the infinite loop:
               #While in the regular RelVal use-case it makes sense to use the actual number of cores of the machines, in
               #the IB case the AvailableCores will always consist of only 1 single core..
               OriginalAvailableCores=list(AvailableCores) #Tricky list copy bug! without the list() OriginalAvalaibleCores would point to AvailableCores!
               #Print this out in the log for debugging reasons
               self.printFlush("Original available cores list: %s"%AvailableCores)

               #Create a dictionary to keep track of running threads on the various cores:
               activePerfTestThreads={}
               #Flag for waiting messages:
               Waiting=False
               while 1:
                  #Check if there are tests to run:
                  if TestsToDo:
                     #Using the Waiting flag to avoid writing this message every 5 seconds in the case
                     #of having more tests to do than available cores...
                     if not Waiting:
                        self.printFlush("Currently %s tests are scheduled to be run:"%len(TestsToDo))
                        self.printFlush(TestsToDo)
                     #Check the available cores:
                     if AvailableCores:
                        #Set waiting flag to False since we'll be doing something
                        Waiting=False
                        self.printFlush("There is/are %s core(s) available"%len(AvailableCores))
                        cpu=AvailableCores.pop()
                        self.printFlush("Let's use cpu %s"%cpu)
                        simpleGenReportArgs=TestsToDo.pop()
                        self.printFlush("Let's submit %s test on core %s"%(simpleGenReportArgs['Name'],cpu))
                        #Adding a Total timer for each of the threaded tests:
                        if simpleGenReportArgs['Name'] not in TimerInfo.keys():
                           #if 'TotalTime' not in TimerInfo[simpleGenReportArgs['Name']].keys():
                           self.PerfTestTotalTimer=PerfSuiteTimer(start=datetime.datetime.now()) #Start the TimeSize timer
                           TimerInfo.update({simpleGenReportArgs['Name']:{'TotalTime':self.PerfTestTotalTimer}}) #Add the TimeSize timer to the dictionary 
                        threadToDo=self.simpleGenReportThread(cpu,self,**simpleGenReportArgs) #Need to send self too, so that the thread has access to the PerfSuite.simpleGenReport() function
                        self.printFlush("Starting thread %s"%threadToDo)
                        ReportExitCode=threadToDo.start()
                        self.printFlush("Adding thread %s to the list of active threads"%threadToDo)
                        activePerfTestThreads[cpu]=threadToDo
                     #If there is no available core, pass, there will be some checking of activeThreads, a little sleep and then another check.
                     else:
                        pass
                  #Test activePerfThreads:
                  activeTestNames=[]
                  activeTestNamesPU=[]
                  for cpu in activePerfTestThreads.keys():
                     if activePerfTestThreads[cpu].isAlive():
                        #print "%% cpu %s activerPerfTestThreads[cpu] %s activePerfTestThreads[cpu].simpleGenReportArgs['cmsdriverOptions'] %s"%(cpu,activePerfTestThreads[cpu],activePerfTestThreads[cpu].simpleGenReportArgs['cmsdriverOptions'])
                        if "--pileup" in activePerfTestThreads[cpu].simpleGenReportArgs['cmsdriverOptions']:
                           activeTestNamesPU.append(activePerfTestThreads[cpu].simpleGenReportArgs['Name'])
                        else:
                           activeTestNames.append(activePerfTestThreads[cpu].simpleGenReportArgs['Name'])
                        pass
                     elif cpu not in AvailableCores:
                        #Set waiting flag to False since we'll be doing something
                        Waiting=False
                        self.printFlush(time.ctime())
                        self.printFlush("%s test, in thread %s is done running on core %s"%(activePerfTestThreads[cpu].simpleGenReportArgs['Name'],activePerfTestThreads[cpu],cpu) )
                        self.printFlush("About to append cpu %s to AvailableCores list"%cpu)
                        AvailableCores.append(cpu)
                        #Eliminate from activeTestNames lists:
                        #print activeTestNames
                        #print activeTestNamesPU
                        #print activePerfTestThreads[cpu].simpleGenReportArgs['Name']
                        if "--pileup" in activePerfTestThreads[cpu].simpleGenReportArgs['cmsdriverOptions']:
                           try:
                              activeTestNamesPU.remove(activePerfTestThreads[cpu].simpleGenReportArgs['Name'])
                           except:
                              pass
                        else:
                           try:
                              activeTestNames.remove(activePerfTestThreads[cpu].simpleGenReportArgs['Name'])
                           except:
                              pass
                        #Eliminate also from activePErfTestThreads dictionary:
                        activePerfTestThreads.pop(cpu)
                        #FIXME:
                        #Delicate check to stop the timer on the individual threaded test!
                        #Need to thik about it still...
                  #FIXME:
                  #Delicate check to stop the timers on the threaded tests:
                  #Check activePerfTestThreads dictionary for "Name" if any name is missing, the total can be stopped for that name.
                  #self.PerfTestTotalTimer
                  for TestName in ["IgProf_Perf","IgProf_Mem","Memcheck","Valgrind"]:
                     if (TestName not in activeTestNames) and (TestName not in activeTestNamesPU) :
                        try:
                           TimerInfo[TestName]['TotalTime'].set_end(datetime.datetime.now())
                        except:
                           #print "No %s test was running"%TestName
                           pass
                  #Buggy if... it seems we don't wait for the running thread to be finished...
                  #We should request:
                  #-All OriginalAvailableCores should be actually available.
                  if not AvailableCores==[] and (set(AvailableCores)==set(range(cmsCpuInfo.get_NumOfCores())) or set(AvailableCores)==set(OriginalAvailableCores)) and not TestsToDo:
                     self.printFlush("PHEW! We're done... all TestsToDo are done... at %s "%(self.getDate()))
                     #Debug printouts:
                     #print "AvailableCores",AvailableCores
                     #print "set(AvailableCores)",set(AvailableCores)
                     #print "set(range(cmsCpuInfo.get_NumOfCores())",set(range(cmsCpuInfo.get_NumOfCores()))
                     #print "OriginalAvailableCores",OriginalAvailableCores
                     #print "set(OriginalAvailableCores)",set(OriginalAvailableCores)                                   
                     #print "TestsToDo",TestsToDo
                     break
                  else:
                     #Putting the sleep statement first to avoid writing Waiting... before the output of the started thread reaches the log... 
                     time.sleep(5)
                     #Use Waiting flag to writing 1 waiting message while waiting and avoid having 1 message every 5 seconds...
                     if not Waiting:
                        self.printFlush(time.ctime())
                        self.printFlush("Waiting for tests to be done...")
                        sys.stdout.flush()
                        Waiting=True
            #End of the if for IgProf, Callgrind, Memcheck tests      
                  
            if benching and not (self._unittest or self._noexec):
                #Ending the performance suite with the cmsScimark benchmarks again:
                for cpu in cpus:
                    if cmsScimark > 0:
                        self.logh.write("Ending with %s cmsScimark on cpu%s\n"         % (cmsScimark,cpu))
                        cmsScimarkFinalTime=PerfSuiteTimer(start=datetime.datetime.now()) #Create the cmsScimark PerfSuiteTimer
                        TimerInfo['cmsScimarkTime'].update({'cmsScimarkFinal':cmsScimarkFinalTime}) #Add the cmsScimarkFinalTime information to the general TimerInfo dictionary

                        self.benchmarks(cpu,perfsuitedir,scimark.name,cmsScimark)
                        cmsScimarkFinalTime.set_end(datetime.datetime.now()) #Stop the cmsScimarkLarge Initial timer
                    if cmsScimarkLarge > 0:
                        self.logh.write("Following with %s cmsScimarkLarge on cpu%s\n" % (cmsScimarkLarge,cpu))
                        cmsScimarkLargeFinalTime=PerfSuiteTimer(start=datetime.datetime.now()) #Create the cmsScimarkLargePerfSuiteTimer
                        TimerInfo['cmsScimarkTime'].update({'cmsScimarkLargeFinal':cmsScimarkLargeFinalTime}) #Add the cmsScimarkLargeFinalTime information to the general TimerInfo dictionary
                        self.benchmarks(cpu,perfsuitedir,scimarklarge.name,cmsScimarkLarge,large=True)
                        cmsScimarkLargeFinalTime.set_end(datetime.datetime.now()) #Stop the cmsScimarkLarge Initial timer
    
            if prevrel:
                self.logh.write("Running the regression analysis with respect to %s\n"%getVerFromLog(prevrel))
                self.logh.write(time.ctime(time.time()))
                self.logh.flush()
                
                crr.regressReports(prevrel,os.path.abspath(perfsuitedir),oldRelName = getVerFromLog(prevrel),newRelName=self.cmssw_version)
    
            #Create a tarball of the work directory
            if tarball:
               tarballTime=PerfSuiteTimer(start=datetime.datetime.now()) #Create the tarball PerfSuiteTimer
               TimerInfo.update({'tarballTime':{'TotalTime':tarballTime}})
               # Adding the str(stepOptions to distinguish the tarballs for 1 release
               # (GEN->DIGI, L1->RECO will be run in parallel)
               
               # Cleaning the stepOptions from the --usersteps=:
               if "=" in str(stepOptions):
                  fileStepOption=str(stepOptions).split("=")[1]
               else:
                  fileStepOption=str(stepOptions)
               if fileStepOption=="":
                  fileStepOption="UnknownStep"
               # Add the working directory used to avoid overwriting castor files (also put a check...)
               fileWorkingDir=os.path.basename(perfsuitedir)
               
               # Also add the --conditions and --eventcontent options used in the --cmsdriver options since it
               # is possible that the same tests will be run with different conditions and/or event content:               
               # Parse it out of --cmsdriver option:
               fileEventContentOption="UnknownEventContent"
               fileConditionsOption="UnknownConditions"
               for token in cmsdriverOptions.split("--"):
                  if token!='' and 'cmsdriver' not in token:
                     if "=" in token:
                        fileOption=token.split("=")[0]
                        fileOptionValue=token.split("=")[1].strip("'").strip('"')
                     else:
                        fileOption=token.split()[0]
                        fileOptionValue=token.split()[1].strip("'").strip('"')
                     if "eventcontent" or "conditions" in fileOption:
                        if "eventcontent" in fileOption:
                           fileEventContentOption=fileOptionValue
                        elif "conditions" in fileOption:
                           # check if we are using the autoCond style of flexible conditions
                           # if so, expand the condition here so that the file names contain the real conditions
                           if "auto:" in fileOptionValue: 
                              from Configuration.AlCa.autoCond import autoCond
                              fileConditionsOption = autoCond[ fileOptionValue.split(':')[1] ]
                           else:
                              # "old style" conditions, hardcoded values ...
                              # FIXME:
                              # Should put at least the convention in cmsPerfCommons to know how to parse it...
                              # Potential weak point if the conditions tag convention changes...
                              if "," in fileOptionValue: #Since 330, conditions don't have FrontierConditions_GlobalTag, in front of them anymore...
                                 fileConditionsOption=fileOptionValue.split("::")[0].split(",")[1] #"Backward" compatibility
                              else:
                                 fileConditionsOption=fileOptionValue.split("::")[0] 
                  else: # empty token
                     #print "Print this is the token: %s"%token
                     pass
                  
               #self.printFlush("Conditions label to add to the tarball name is %s"%fileConditionsOption)
               #self.printFlush("Eventcontent label to add to the tarball name is %s"%fileEventContentOption)
                     #FIXME:
                     #Could add the allowed event contents in the cmsPerfCommons.py file and use those to match in the command line options... This assumes maintenance of cmsPerfCommons.py


               #Create a tarball with just the logfiles
               subprocess.Popen("ls -R | grep .root > rootFiles",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.read()
               LogFile = "%s_%s_%s_%s_%s_%s_%s_%s_log.tgz" % (self.cmssw_arch, self.cmssw_version, fileStepOption, fileConditionsOption, fileEventContentOption.split()[0], fileWorkingDir, self.host, self.user)
               AbsTarFileLOG = os.path.join(perfsuitedir,LogFile)
               tarcmd  = "tar zcfX %s %s %s" %(AbsTarFileLOG, "rootFiles", os.path.join(perfsuitedir,"*"))
               self.printFlush("Creating a tarball for the logfiles")
               self.printFlush(tarcmd)
               self.printFlush(subprocess.Popen(tarcmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.read())
               self.printFlush(subprocess.Popen("rm rootFiles",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.read())               

               fullcastorpathlog=os.path.join(castordir,LogFile)


               #Create the tarball with the contents of the directory + md5 checksum
               TarFile = "%s_%s_%s_%s_%s_%s_%s_%s.tgz" % (self.cmssw_arch, self.cmssw_version, fileStepOption, fileConditionsOption, fileEventContentOption.split()[0], fileWorkingDir, self.host, self.user)
               AbsTarFile = os.path.join(perfsuitedir,TarFile)
               tarcmd  = "tar -zcf %s %s" %(AbsTarFile, os.path.join(perfsuitedir,"*"))
               md5cmd = "md5sum %s" %(AbsTarFile)
               self.printFlush("Creating a tarball with the content of the directory")               
               self.printFlush(tarcmd)
               self.printFlush(md5cmd)               
               #FIXME:
               #Anything that will be logged after the tar command below will not enter the cmsPerfSuite.log in the tarball (by definition)...
               #To remain backward compatible the harvesting script needs to be based on the command above to identify the tarball location.
               #Obsolete popen4-> subprocess.Popen
               #self.printFlush(os.popen3(tarcmd)[2].read()) #Using popen3 to get only stderr we don't want the whole stdout of tar!
               self.printFlush(subprocess.Popen(tarcmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.read())
               md5sum = subprocess.Popen(md5cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.read().split()[0]
               self.printFlush("The md5 checksum of the tarball: %s" %(md5sum))
               AbsTarFileMD5 = AbsTarFile + ".md5"
               md5filecmd = "echo %s > %s" % (md5sum, AbsTarFileMD5)
               self.printFlush(subprocess.Popen(md5filecmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.read())
                                                
               #Archive it on CASTOR
               #Before archiving check if it already exist if it does print a message, but do not overwrite, so do not delete it from local dir:
               fullcastorpathfile=os.path.join(castordir,TarFile)
               fullcastorpathmd5=os.path.join(castordir,TarFile + ".md5")
               
               checkcastor="nsls  %s" % fullcastorpathfile
               #Obsolete os.popen-> subprocess.Popen                
               #checkcastorout=os.popen3(checkcastor)[1].read()
               checkcastorout=subprocess.Popen(checkcastor,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.read()                
               if checkcastorout.rstrip()==fullcastorpathfile:
                  castorcmdstderr="File %s is already on CASTOR! Will NOT OVERWRITE!!!"%fullcastorpathfile
               else:
                  #Switching from CASTOR TO EOS, i.e. rfcp to xrdcp!
                  #Not YET!!!
                  #FIXME! Migrate to EOS eventually, taking into account implications for PerfDB logs linking!
                  castorcmd="rfcp %s %s" % (AbsTarFile,fullcastorpathfile)
                  castormd5cmd="rfcp %s %s" % (AbsTarFileMD5,fullcastorpathmd5)
                  castorlogcmd="rfcp %s %s" % (AbsTarFileLOG,fullcastorpathlog)
                  self.printFlush(castorcmd)
                  self.printFlush(castormd5cmd)
                  self.printFlush(castorlogcmd)
                  #Obsolete os.popen-> subprocess.Popen
                  #castorcmdstderr=os.popen3(castorcmd)[2].read()
                  castorcmdstderr=subprocess.Popen(castorcmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.read()
                  subprocess.Popen(castormd5cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.read()
                  subprocess.Popen(castorlogcmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr.read()                  
               #Checking the stderr of the rfcp command to copy the tarball (.tgz) on CASTOR:
               if castorcmdstderr:
                   #If it failed print the stderr message to the log and tell the user the tarball (.tgz) is kept in the working directory
                   self.printFlush(castorcmdstderr)
                   self.printFlush("Since the CASTOR archiving for the tarball failed the file %s is kept in directory %s"%(TarFile, perfsuitedir))
               else:
                   #If it was successful then remove the tarball from the working directory:
                   self.printFlush("Successfully archived the tarball %s in CASTOR!"%(TarFile))
                   self.printFlush("The tarball can be found: %s"%(fullcastorpathfile))
                   self.printFlush("The logfile can be found: %s"%(fullcastorpathlog))                   
                   self.printFlush("Deleting the local copy of the tarballs")
                   rmtarballcmd="rm -Rf %s"%(AbsTarFile)
                   rmtarballmd5cmd="rm -Rf %s"%(AbsTarFileMD5)
                   rmtarballlogcmd="rm -Rf %s"%(AbsTarFileLOG)
                   self.printFlush(rmtarballcmd)
                   self.printFlush(rmtarballmd5cmd)
                   self.printFlush(rmtarballlogcmd)                   
                   #Obsolete os.popen-> subprocess.Popen
                   #self.printFlush(os.popen4(rmtarballcmd)[1].read())
                   self.printFlush(subprocess.Popen(rmtarballcmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read() )
                   self.printFlush(subprocess.Popen(rmtarballmd5cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read() )
                   self.printFlush(subprocess.Popen(rmtarballlogcmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read() )
               tarballTime.set_end(datetime.datetime.now())
            else:
               self.printFlush("Performance Suite directory will not be archived in a tarball since --no_tarball option was chosen")

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
        #Add the possibility to send as an email the execution logfile to the user and whoever else interested:
        if MailLogRecipients != "": #Basically leave the option to turn it off too.. --mail ""
           self.printFlush("Sending email notification for this execution of the performance suite with command:")
           sendLogByMailcmd='cat cmsPerfSuite.log |mail -s "Performance Suite finished running on %s" '%self.host + MailLogRecipients
           self.printFlush(sendLogByMailcmd)
           #Obsolete os.popen-> subprocess.Popen
           #self.printFlush(os.popen4(sendLogByMailcmd)[1].read())
           self.printFlush(subprocess.Popen(sendLogByMailcmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read() )
        else:
           self.printFlush('No email notification will be sent for this execution of the performance suite since option --mail "" was used')
        
        TotalTime.set_end(datetime.datetime.now())        
        self.printFlush("Total Running Time\t%s hrs (%s mins)"%(TotalTime.get_duration()['hours'],TotalTime.get_duration()['minutes']))

        #Dump of the TimerInfo information
        #First dump it as a pickleq file...
        #Well in order to do so without the complication of serializing a custom class instance need to make the dictionary fully string-made:
        TimerInfoStr={}
        PerfSuiteTimerInfo=open("PerfSuiteTimerInfo.pkl","wb")
        #pickle.dump(TimerInfo,PerfSuiteTimerInfo)
        #PerfSuiteTimerInfo.close()
        #For now print it at the bottom of the log:
        self.logh.write("Test type\tActual Test\tDuration\tStart Time\tEnd Time\n")
        for key in TimerInfo.keys():
           #self.printFlush(key)
           TimerInfoStr.update({key:{}})
           for test in TimerInfo[key].keys():
              TimerInfoStr[key].update({test:[str(TimerInfo[key][test].get_duration()['hours'])+" hrs ("+str(TimerInfo[key][test].get_duration()['minutes'])+" mins)",TimerInfo[key][test].get_start(),TimerInfo[key][test].get_end()]})
              self.logh.write(key+"\t"+test+"\t")
              self.logh.write("%s hrs (%s mins)\t"%(TimerInfo[key][test].get_duration()['hours'],TimerInfo[key][test].get_duration()['minutes']))
              self.logh.write("%s\t"%TimerInfo[key][test].get_start())
              self.logh.write("%s\n"%TimerInfo[key][test].get_end())
        pickle.dump(TimerInfoStr,PerfSuiteTimerInfo) 
        PerfSuiteTimerInfo.close()
        
        self.logh.write("Final Performance Suite exit code was %s"%FinalExitCode)
        self.logh.flush()
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

    #print suite
    #Uncomment this for tests with main() in inteactive python:
    #print suite.optionParse(argv)
    
    PerfSuiteArgs={}
    (PerfSuiteArgs['create'],
     PerfSuiteArgs['castordir'],
     PerfSuiteArgs['TimeSizeEvents'],
     PerfSuiteArgs['IgProfEvents'],    
     PerfSuiteArgs['CallgrindEvents'],
     PerfSuiteArgs['MemcheckEvents'],
     PerfSuiteArgs['cmsScimark'],      
     PerfSuiteArgs['cmsScimarkLarge'], 
     PerfSuiteArgs['cmsdriverOptions'],
     PerfSuiteArgs['cmsdriverPUOptions'],
     PerfSuiteArgs['stepOptions'],     
     PerfSuiteArgs['quicktest'],       
     PerfSuiteArgs['profilers'],       
     PerfSuiteArgs['cpus'],            
     PerfSuiteArgs['cores'],           
     PerfSuiteArgs['prevrel'],         
     PerfSuiteArgs['bypasshlt'],       
     PerfSuiteArgs['runonspare'],      
     PerfSuiteArgs['perfsuitedir'],    
     PerfSuiteArgs['logfile'],
     PerfSuiteArgs['TimeSizeCandles'],
     PerfSuiteArgs['IgProfCandles'],
     PerfSuiteArgs['CallgrindCandles'],
     PerfSuiteArgs['MemcheckCandles'],
     PerfSuiteArgs['TimeSizePUCandles'],
     PerfSuiteArgs['IgProfPUCandles'],
     PerfSuiteArgs['CallgrindPUCandles'],
     PerfSuiteArgs['MemcheckPUCandles'],
     PerfSuiteArgs['PUInputFile'],
     PerfSuiteArgs['userInputFile'],
     PerfSuiteArgs['MailLogRecipients'],
     PerfSuiteArgs['tarball']
     ) = suite.optionParse(argv)

    if PerfSuiteArgs['create']: # Before anything, request the AFS volume (it takes some time...)
       suite.createIgVolume()

    if not PerfSuiteArgs['logfile'] == None:
       if os.path.exists(PerfSuiteArgs['logfile']):
          oldlogfile=PerfSuiteArgs['logfile']+"_"+time.strftime("%d-%m-%Y_%H:%M:%S")
          #Move old logfile to a file with the same filename plus a timestamp appended
          mvOldLogfile=subprocess.Popen("mv %s %s"%(PerfSuiteArgs['logfile'],oldlogfile), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
          mvOldLogfileExitCode=mvOldLogfile.wait()
          #Finally open the logfile and put the information above in it:
          try:
             ActualLogfile = open(PerfSuiteArgs['logfile'],"w")
             if mvOldLogfileExitCode:
                ActualLogfile.write("Please check what happened: A file named %s existed already and the attempt to move it to %s produced the following output: %s\n"%(PerfSuiteArgs['logfile'],oldlogfile,mvOldLogfile.stdout))
             else:
                ActualLogfile.write("***WARNING! A file named %s existed already!\n***It has been moved to %s before starting the current logfile!\n"%(PerfSuiteArgs['logfile'],oldlogfile))
          except (OSError, IOError), detail:
             ActualLogfile.write("Failed to open the intended logfile %s, detail error:\n%s"%(PerfSuiteArgs['logfile'],detail))
              
       else:
          try:
             ActualLogfile = open(PerfSuiteArgs['logfile'],"w")
          except (OSError, IOError), detail:
             ActualLogfile.write("Failed to open the intended logfile %s, detail error:\n%s"%(PerfSuiteArgs['logfile'],detail))
       ActualLogfile.flush()
                 
    #Three lines to add the exact command line used to call the performance suite directly in the log.
    ActualLogfile.write("Performance suite invoked with command line:\n")
    cmdline=reduce(lambda x,y:x+" "+y,sys.argv)
    ActualLogfile.write(cmdline+"\n")
    ActualLogfile.flush()
    
    #Debug printout that we could silence...
    ActualLogfile.write("Initial PerfSuite Arguments:\n")
    for key in PerfSuiteArgs.keys():
        ActualLogfile.write("%s %s\n"%(key,PerfSuiteArgs[key]))
    ActualLogfile.flush()
    #print PerfSuiteArgs

    PerfSuiteArgs['cpu_list'] = PerfSuiteArgs['cpus'] #To access the actual number of cpus used inside the threads..

    #Handle in here the case of multiple cores and the loading of cores with cmsScimark:
    if len(PerfSuiteArgs['cpus']) > 1:
        ActualLogfile.write("More than 1 cpu: threading the Performance Suite!\n")
        outputdir=PerfSuiteArgs['perfsuitedir']
        runonspare=PerfSuiteArgs['runonspare'] #Save the original value of runonspare for cmsScimark stuff
        cpus=PerfSuiteArgs['cpus']
        cores=PerfSuiteArgs['cores']
        if runonspare:
            for core in range(PerfSuiteArgs['cores']):
                cmsScimarkLaunch_pslist={}
                if len(cpus) != cores: #In case of this (relval), don't load the others with cmsScimark
                   if (core not in cpus):
                      #self.logh.write("Submitting cmsScimarkLaunch.csh to run on core cpu "+str(core) + "\n")
                      ActualLogfile.write("Submitting cmsScimarkLaunch.csh to run on core cpu "+str(core)+"\n")
                      subcmd = "cd %s ; cmsScimarkLaunch.csh %s" % (outputdir, str(core))            
                      command="taskset -c %s sh -c \"%s\" &" % (str(core), subcmd)
                      #self.logh.write(command + "\n")
                      ActualLogfile.write(command+"\n")
                      #cmsScimarkLaunch.csh is an infinite loop to spawn cmsScimark2 on the other
                      #cpus so it makes no sense to try reading its stdout/err
                      cmsScimarkLaunch_pslist[core]=subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                      ActualLogfile.write("Spawned %s \n with PID %s"%(command,cmsScimarkLaunch_pslist[core].pid))
                      ActualLogfile.flush()
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
            #ActualLogfile.write(suitethread[cpu])
            ActualLogfile.write("Launching PerfSuite thread on cpu%s"%cpu)
            ActualLogfile.flush()
            #print "With arguments:"
            #print PerfSuiteArgs
            suitethread[cpu].start()
            
        while reduce(lambda x,y: x or y, map(lambda x: x.isAlive(),suitethread.values())):
           try:            
              time.sleep(5.0)
              sys.stdout.flush()
           except (KeyboardInterrupt, SystemExit):
              raise
        ActualLogfile.write("All PerfSuite threads have completed!\n")
        ActualLogfile.flush()

    else: #No threading, just run the performance suite on the cpu core selected
        suite.runPerfSuite(**PerfSuiteArgs)
    
if __name__ == "__main__":
    
    main(sys.argv)
