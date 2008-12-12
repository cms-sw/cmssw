#!/usr/bin/env python
# -*- coding: utf-8 -*-
# GBenelli Dec 21, 2007 and JNicolson July 23, 2008
# This script is designed to run on a local directory
# after the user has created a local CMSSW release,
# initialized its environment variables by executing
# in the release /src directory:
# eval `scramv1 runtime -csh`
# project CMSSW
# The script will create a SimulationCandles_$CMSSW_VERSION.txt ASCII
# file, input to cmsRelvalreport.py, to launch the
# Performance Suite.

# Input arguments are three:
# 1-Number of events to put in the cfg files
# 2-Name of the candle(s) to process (either AllCandles, or NameOfTheCandle)
# 3-Profiles to run (with code below)
# E.g.: ./cmsRelValReportInput.py 50 AllCandles 012

###############
# Import modules and define functions
#

import sys, os, re, operator
import optparse as opt
from cmsPerfCommons import Candles, MIN_REQ_TS_EVENTS, CandDesc, FileName, KeywordToCfi, CustomiseFragment, CandFname
################
# Global variables
#

THIS_PROG_NAME = os.path.basename(sys.argv[0])
cmsDriver = 'cmsDriver.py'                    #cmsDriver.py path
hypreg = re.compile('-')
debug = False
DEF_STEPS = ['GEN,SIM', 'DIGI']
AllSteps  = ["GEN,SIM", "DIGI", "L1", "DIGI2RAW", "HLT", "RAW2DIGI","RECO"]
AfterPileUpSteps=[]

# Global variables used by writeCommandsToReport and dependents

# Hash for the profiler to run

Profiler = {
    'TimingReport'            : 'Timing_Parser',
    'TimingReport @@@ reuse'  : 'Timing_Parser',
    'TimeReport'              : 'Timereport_Parser',
    'TimeReport @@@ reuse'    : 'Timereport_Parser',
    'SimpleMemReport'         : 'SimpleMem_Parser',
    'EdmSize'                 : 'Edm_Size',
    'IgProfperf'              : 'IgProf_perf.PERF_TICKS',
    'IgProfMemTotal'          : 'IgProf_mem.MEM_TOTAL',
    'IgProfMemTotal @@@ reuse': 'IgProf_mem.MEM_TOTAL',
    'IgProfMemLive'           : 'IgProf_mem.MEM_LIVE',
    'IgProfMemLive @@@ reuse' : 'IgProf_mem.MEM_LIVE',
    'IgProfMemAnalyse'        : 'IgProf_mem.ANALYSE',
    'valgrind'                : 'ValgrindFCE',
    'memcheck_valgrind'       : 'Memcheck_Valgrind',
    'None'                    : 'None',
}



def getFstOccur(item, list):
    return filter(item.__eq__,list)[0]

def getLstIndex(item, list):
    lenlist = len(list)
    for x in range(lenlist - 1,0,-1):
        if list[x] == item:
            return x

def checkSteps(steps):
    idx = -2
    lstidx = -2
    for step in steps:
        astep = step
        split = []
        if "-" in step:
            split = astep.split("-")
            astep = split[0]
        idx = AllSteps.index(astep)
        if not ( idx == -2 ):
            if lstidx > idx:
                print "ERROR: Your user defined steps are not in a valid order"
                sys.exit()
        lstidx = idx    
        if "-" in step:
            lstidx = AllSteps.index(split[1])


def getSteps(userSteps):

    # Then split the user steps into "steps"
    gsreg = re.compile('GEN-SIM')
    greg = re.compile('GEN') #Add a second hack (due to the first) to handle the step 1 case GEN-HLT
    StepsTokens = userSteps.split(",")
    steps = [] 
    for astep in StepsTokens:

        # Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO)
        # from using the "-" to using the "," to match cmsDriver.py convention

        if gsreg.search(astep):
            astep = gsreg.sub(r"GEN,SIM", astep)
        elif greg.search(astep):
            astep = greg.sub(r"GEN,SIM", astep)
            
        print astep
        # Finally collect all the steps into the @Steps array:

        steps.append(astep)
    
    #steps = expandHypens(steps)
    checkSteps(steps)
    return steps
        
def optionparse():
    global _noprof
    explanations = map(lambda x: "          " + x, Candles)
    explanation  = ""
    for x in range(len(explanations)):
        explanation += "%-*s %s\n" % (30, explanations[x],CandDesc[x])
    parser = opt.OptionParser(usage=("""%s NUM_EVENTS_PER_CFG CANDLES PROFILE [--cmsdriver=cmsDriverOptions] [--usersteps=processingStepsOption]

    Description - This program creates a configuration file for cmsRelvalreport.py that describes the order in which cmsDriver.py should be run with which candles, steps and profiling so that time spent running cmsDriver.py is minimised. Without this correct ordering we would have to re-run parts of the profiling steps.

    Arguments:
        NUM_EVENTS_PER_CFG - The number of events per config file

        CANDLES - The simulation type to perform profiling on
          Candles              Description
          AllCandles           Run all of the candles below
%s

        PROFILE - the type of profiling to perform (multiple codes can be used):
          Code  Profile Type
          0     TimingReport
          1     TimeReport
          2     SimpleMemoryCheck
          3     EdmSize
          4     IgProfPerf
          5     IgProfMemTotal
          6     IgProfMemLive
          7     IgProfAnalyse
          8     ValgrindFCE
          9     ValgrindMemCheck    

    Examples:  
       Perform Timing Report profiling for all candles and 10 events per cfg.
        ./%s 10 AllCandles 1
       Perform Timing Report, Time Report and SimpleMemoryCheck profiling for Higgs boson and 50 events.
        ./%s 50 \"HZZLLLL\" 012
       Perform IgProfPerf and IgProfMemTotal profiling for TTbar and 100 events.
        ./%s 100 \"TTBAR\" 45 --cmsdriver=\"--conditions FakeConditions\"
       Perform ValgrindFCE ValgrindMemCheck profiling for Minimum bias and 100 events. Only on gensim and digi steps.
        ./%s 100 \"MINBIAS\" 89 --cmsdriver=\"--conditions FakeConditions\" \"--usersteps=GEN-SIM,DIGI"""
      % ( THIS_PROG_NAME, explanation, THIS_PROG_NAME,THIS_PROG_NAME,THIS_PROG_NAME,THIS_PROG_NAME)))
    
    devel  = opt.OptionGroup(parser, "Developer Options",
                                     "Caution: use these options at your own risk."
                                     "It is believed that some of them bite.\n")
    #parser.set_defaults(debug=False)

    parser.add_option(
        '-b',
        '--bypass-hlt',
        action="store_true",
        dest='bypasshlt',
        default=False,
        help='Should we bypass using the HLT root file?'
        )

    parser.add_option(
        '-u',
        '--usersteps',
        type='string',
        dest='userSteps',
        help='Which steps to run',
        metavar='<STEPS>',
        )
    parser.add_option(
        '-c',
        '--cmsdriver',
        type='string',
        dest='cmsDriverOptions',
        help='Option for cmsDriver.py can be specified as a string to be added to all cmsDriver.py commands',
        metavar='<CONDITION>',
        )

    devel.add_option(
        '-1',
        '--no-profile',
        action="store_true",
        dest='noprof',
        help='Do not perform profiling, ever',
        #metavar='DEBUG',
        )
    
    devel.add_option(
        '-d',
        '--debug',
        action="store_true",
        dest='debug',
        help='Show debug output',
        #metavar='DEBUG',
        )

    parser.set_defaults(debug=False,noprof=False)
    parser.add_option_group(devel)

    (options, args) = parser.parse_args()
    

    _noprof = options.noprof
    debug = options.debug
    numofargs = len(args) 
#    print "runtime cmsinput " + str(numofargs) + " noprof "  + str(_noprof)
    if ((not numofargs == 3) and (not _noprof)) or (_noprof and not (numofargs == 2)):
        parser.error("There are not enough arguments specified to run this program."
                     " Please determine the correct arguments from the usage information above."
                     " Run with --help option for more information.")
        sys.exit()

    return (options, args)

def expandHyphens(step):
    newsteps = []
#    for step in steps:
    if "-" in step:
        hypsteps = step.split(r"-")
        if not (len(hypsteps) == 2):
            print "ERROR: Could not parse usersteps. You have too many hypens between commas"
            sys.exit()
        elif not (hypsteps[0] in AllSteps and hypsteps[1] in AllSteps):
            print "ERROR: One of the steps you defined is invalid"
            sys.exit()
        else:
            if (hypsteps[0] == hypsteps[1]):
                print "WARNING: You should not add a hypenated step that as the same source and destination step, ignoring"
                newsteps.append(hypsteps[0])
            else:
                newsteps.append(hypsteps[0])
                srt = AllSteps.index(hypsteps[0]) + 1
                for n in range(srt,len(AllSteps) - 1,1):
                    astep = AllSteps[n]
                    if astep == hypsteps[1]:
                        break
                    else:
                        newsteps.append(astep)
                newsteps.append(hypsteps[1])
    else:
        if not (step in AllSteps):
            print "ERROR: One of the steps you defined is invalid"
            sys.exit()
        else:
            newsteps.append(step)

    return newsteps

def setupProgramParameters(options,args):
    steps = []
    cmsDriverOptions = ""
    global AfterPileUpSteps
    NumberOfEvents = int(args[0])  # first arg
    WhichCandles   = str(args[1])  # second arg
    ProfileCode = ""
    if not _noprof:
        ProfileCode    = str(args[2])  # third arg

    if options.cmsDriverOptions:

        cmsDriverOptions = options.cmsDriverOptions
        print 'Using user-specified cmsDriver.py options: ' + cmsDriverOptions

    if options.userSteps:

        userSteps = options.userSteps
        steps = getSteps(userSteps)
        #Set the steps to be run after DIGI-PILEUP:
        #FIXME: This is not very robust composite steps could mess up things...
        #but for now let's assume PILE UP is not run in all complicated cases...
        if 'DIGI' in steps:
            AfterPileUpSteps = steps[steps.index('DIGI')+1:] #The DIGI step needs to be included to actually do the pileup!
        else:
            #The following should work for the case L1-RECO and RAW2DIGI-RECO...
            if steps[0] in AllSteps:
                if AllSteps.index(steps[0]) == AllSteps.index('DIGI')+1 : #i.e. if steps[0] is L1...
                    AfterPileUpSteps.append(AllSteps[AllSteps.index(steps[0])])
                else:
                    for StepIndex in range(AllSteps.index('DIGI')+1, AllSteps.index(steps[0])+1):
                        AfterPileUpSteps.append(AllSteps[StepIndex])
            elif expandHyphens(steps[0])[0] in AllSteps:
                
                #Add the following cases:
                #1-Step1 : --step GEN-HLT ->Run the DIGI-PU and then all steps from L1 to HLT.
                #2-Step2: --RAW2DIGI-RECO (keep in mind it could change so make this configurable in cmsPErfCommons.py ->run DIGI-PU and then  
                #3-GEN-SIM,DIGI
                #4-Userdefined in cmsPerfCommons.py
                #Other way we need to handle the hyphens even in here... and open everythin up...
                if AllSteps.index(expandHyphens(steps[0])[0]) == AllSteps.index('DIGI')+1 : #i.e. if steps[0] is L1...
                    AfterPileUpSteps.append(AllSteps[AllSteps.index(expandHyphens(steps[0])[0])])
                else:
                    for StepIndex in range(AllSteps.index('DIGI')+1, AllSteps.index(expandHyphens(steps[0])[0])+1):
                        if AllSteps[StepIndex]=='RAW2DIGI':#Dirty hardcoded stuff to handle RAW2DIGI-RECO as one step (should we just hardcode this in the steps?Any negative onsequences?
                            AfterPileUpSteps.append('RAW2DIGI-RECO')
                        else:
                            AfterPileUpSteps.append(AllSteps[StepIndex])
                if expandHyphens(steps[0])[1] in AllSteps and not expandHyphens(steps[0])[1]=='RECO': #Dirty hardcoded stuff to handle RAW2DIGI-RECO as one step (should we just hardcode this in the steps?Any negative onsequences?
                    AfterPileUpSteps.append(AllSteps[AllSteps.index(expandHyphens(steps[0])[1])])
            AfterPileUpSteps.extend(steps[1:])
            print "After pileup steps: %s"%AfterPileUpSteps

    if WhichCandles.lower() == 'allcandles':
        Candle = Candles
        print 'ALL standard simulation candles will be PROCESSED:'
    else:
        Candle = [WhichCandles]
        print 'ONLY %s will be PROCESSED' % Candle[0]

    # For now the two steps are built in, this can be added as an argument later
    # Added the argument option so now this will only be defined if it was not defined already:

    if not steps:
        print 'The default steps will be run:'
        steps = DEF_STEPS
    else:
        print "You defined your own steps to run:"

    for astep in steps:
        print astep

    return (NumberOfEvents, ProfileCode, cmsDriverOptions, steps, Candle, options.bypasshlt)

def init_vars():

    ####################
    # Obtain required environment variables
    #

    try:
        CMSSW_BASE         = os.environ['CMSSW_BASE']
        CMSSW_RELEASE_BASE = os.environ['CMSSW_RELEASE_BASE']
        CMSSW_VERSION      = os.environ['CMSSW_VERSION']
    except KeyError:
        print 'Error: An environment variable either CMSSW_{BASE, RELEASE_BASE or VERSION} is not available.'
        print '       Please run eval `scramv1 runtime -csh` to set your environment variables'
        sys.exit()

    return ( CMSSW_BASE,
             CMSSW_RELEASE_BASE,
             CMSSW_VERSION)

def writeReportFileHeader(CMSSW_VERSION,CMSSW_RELEASE_BASE,CMSSW_BASE):

    SimCandlesFile = 'SimulationCandles_%s.txt' % CMSSW_VERSION
    
    try:
        simcandles = open(SimCandlesFile, 'w')
    except IOError:
        print "Couldn't open %s to save" % SimCandlesFile

    simcandles.write('#Candles file automatically generated by %s for %s\n'
                      % (THIS_PROG_NAME, CMSSW_VERSION))
    simcandles.write("#CMSSW Base  : %s\n"   % CMSSW_BASE)
    simcandles.write("#Release Base: %s\n"   % CMSSW_RELEASE_BASE)
    simcandles.write("#Version     : %s\n\n" % CMSSW_VERSION)

    return simcandles

def getProfileArray(ProfileCode):

    Profile = []

    # The allowed profiles are:

    AllowedProfile = [
        'TimingReport',
        'TimeReport',
        'SimpleMemReport',
        'EdmSize',
        'IgProfperf',
        'IgProfMemTotal',
        'IgProfMemLive',
        'IgProfMemAnalyse',
        'valgrind',
        'memcheck_valgrind',
        'None',
        ]

    if _noprof:
        Profile.append(AllowedProfile[-1])
    else:
        for i in range(10):
            if str(i) in ProfileCode:
                firstCase = i == 0 and (str(1) in ProfileCode or str(2) in ProfileCode) or i == 1 and str(2) in ProfileCode
                secCase   = i == 5 and (str(6) in ProfileCode or str(7) in ProfileCode) or i == 6 and str(7) in ProfileCode
                
                if firstCase or secCase:
                    Profile.append(AllowedProfile[i] + ' @@@ reuse')
                else:
                    Profile.append(AllowedProfile[i])
                
    return Profile

def writeStepHead(simcandles,acandle,step):
    simcandles.write('#%s\n' % FileName[acandle])
    simcandles.write('#Step %s\n' % step)
    print step


def determineNewProfile(step,Profile,SavedProfile):
    if 'DIGI2RAW' in step:
        SavedProfile = Profile
        Profile = [ ]
    if 'HLT' in step:
        Profile = SavedProfile

    return (Profile, SavedProfile)

def pythonFragment(step,cmsdriverOptions):
    # Convenient dictionary to map the correct customise Python fragment for cmsDriver.py:
    #It is now living in cmsPerfCommons.py!

 #   CustomiseFragment = {
 #        'GEN,SIM': 'Validation/Performance/TimeMemoryG4Info.py',
 #        'DIGI': 'Validation/Performance/TimeMemoryInfo.py',
 #        'DIGI-PILEUP':'Validation/Performance/MixingModule.py'}
    
    if "--pileup" in cmsdriverOptions:
        return CustomiseFragment['DIGI-PILEUP']
    elif CustomiseFragment.has_key(step):
        return CustomiseFragment[step] 
    else:
        #This is a safe default in any case,
        #the maintenance of the customise python fragments goes into cmsPerfCommons.py
        return CustomiseFragment['DIGI']


def setInputFile(steps,step,acandle,stepIndex,pileup=False,bypasshlt=False):
    InputFileOption = ""
    if pileup and stepIndex == 0:
        InputFileOption = "--filein file:%s_%s" % ( FileName[acandle],"DIGI" )
    else:
        InputFileOption = "--filein file:%s_%s" % ( FileName[acandle],steps[stepIndex - 1] )
        
    if pileup:
        pass
    else :
        if 'GEN,SIM' in step:  # there is no input file for GEN,SIM!
            InputFileOption = ''
        elif   'HLT' in steps[stepIndex - 1] and bypasshlt:

            # Special hand skipping of HLT since it is not stable enough, so it will not prevent
            # RAW2DIGI,RECO from running

            InputFileOption = "--filein file:%s_%s"  % ( FileName[acandle],steps[stepIndex - 2] )

    if not InputFileOption == "" :
        InputFileOption += ".root "

    return InputFileOption

def writeUnprofiledSteps(simcandles,CustomisePythonFragment,cmsDriverOptions,unprofiledSteps,acandle,NumberOfEvents, stepIndex, bypasshlt):
    # reduce(lambda x,y : x + "," + "y",unprofiledSteps)
    stepsStr = ",".join(unprofiledSteps)

    simcandles.write("\n#Run a %s step(s) that has not been selected for profiling but is needed to run the next step to be profiled\n" % (stepsStr))
    #print "unprofiledSteps is %s"%unprofiledSteps
    #print "22acandle is %s"%acandle
    OutputFile = "%s_%s.root" % ( FileName[acandle],unprofiledSteps[-1])
    OutputFileOption = "--fileout=%s" % OutputFile
    #Bug here: should take into account the flag --bypass-hlt instead of assuming hlt should be bypassed
    #This affects the Step1/Step2 running since Step1 will produce an HLT.root file and Step2 should start from there!
    #Adding the argument bypasshlt to the calls...
    InputFileOption = setInputFile(AllSteps,unprofiledSteps[0],acandle,stepIndex - 1,bypasshlt=bypasshlt)

    Command = ("%s %s -n %s --step=%s %s %s --customise=%s %s"
                       % (cmsDriver,
                          KeywordToCfi[acandle],
                          NumberOfEvents,
                          stepsStr,
                          InputFileOption,
                          OutputFileOption,
                          CustomisePythonFragment,
                          cmsDriverOptions) )
    simcandles.write( "%s @@@ None @@@ None\n\n" % (Command))
    return OutputFile

def writePrerequisteSteps(simcandles,steps,acandle,NumberOfEvents,cmsDriverOptions,bypasshlt):
    fstIdx = -1
    if "-" in steps[0]:
        fstIdx = AllSteps.index(steps[0].split("-")[0])
    else:
        fstIdx = AllSteps.index(steps[0])
    CustomisePythonFragment = pythonFragment("GEN,SIM",cmsDriverOptions)
    #print "fstIdx is %s"%fstIdx
    #print "11acandle is %s"%acandle
    OutputFile = writeUnprofiledSteps(simcandles, CustomisePythonFragment, cmsDriverOptions,AllSteps[0:fstIdx],acandle,NumberOfEvents, 0,bypasshlt) 
    return (fstIdx, OutputFile)

def setOutputFileOption(acandle,endstep):
    return "%s_%s.root" % ( FileName[acandle],endstep)

def writeCommands(simcandles,
                  Profile,
                  acandle,
                  steps,
                  NumberOfEvents,
                  cmsDriverOptions,
                  bypasshlt,                  
                  stepIndex = 0,
                  pileup = False):

    OutputStep = ""

    stopIndex = len(steps)
    start = 0

    userSteps = steps
    SavedProfile = []
    fstROOTfile = True
    fstROOTfileStr = ""

    #Handling the case of the first user step not being the first step (GEN,SIM):
    print steps
    if not (steps[0] == AllSteps[0]) and (steps[0].split("-")[0] != "GEN,SIM"):
        #Write the necessary line to run without profiling all the steps before the wanted ones in one shot:
        #print "00acandle is %s"%acandle
        (stepIndex, rootFileStr) = writePrerequisteSteps(simcandles,steps,acandle,NumberOfEvents,cmsDriverOptions,bypasshlt)
        
        #Now take care of setting the indeces and input root file name right for the profiling part...
        if fstROOTfile:
            fstROOTfileStr = rootFileStr
            fstROOTfile = False
        start = -1
        if "-" in steps[0]:
            start = AllSteps.index(steps[0].split("-")[0])
        else:
            start = AllSteps.index(steps[0])
        lst = - 1
        if "-" in steps[-1]:
            lst   = AllSteps.index(steps[-1].split("-")[1]) #here we are assuming that - is always between two steps no GEN-SIM-DIGI, this is GEN-DIGI
        else:
            lst   = AllSteps.index(steps[-1])
        runSteps = AllSteps[start:lst]
        numOfSteps = (lst - start) + 1
        stopIndex = start + numOfSteps
        #Handling the case in which the first user step is the same as the first step (GEN,SIM)
        #elif  not (steps[0] == AllSteps[0]) and (steps[0].split("-")[0] == "GEN"):
    else:
        #Handling the case of the last step being a composite one:
        if "-" in steps[-1]:
            #Set the stop index at the last step of the composite step (WHY???) 
            stopIndex = AllSteps.index(steps[-1].split("-")[1]) + 1
        else:
            stopIndex = AllSteps.index(steps[-1]) + 1
            
    steps = AllSteps
            
    unprofiledSteps = []
    rawreg = re.compile("^RAW2DIGI")
    
#   FOR step in steps

    prevPrevOutputFile = ""
    previousOutputFile = ""

    for x in range(start,stopIndex,1):
        if stepIndex >= stopIndex:
            break
        step = steps[stepIndex]
        # One shot Profiling variables
        befStep     = step
        aftStep     = step
        # We need this in case we are running one-shot profiling or for DIGI-PILEUP
        stepToWrite = step 
        
        CustomisePythonFragment = pythonFragment(step,cmsDriverOptions)
        oneShotProf = False
        hypsteps = []

        #Looking for step in userSteps, or for composite steps that step matches the first of a composite step in userSteps
        if step in userSteps or reduce(lambda x,y : x or y,map(lambda x: step == x.split("-")[0],userSteps)): 

            #Checking now if the current step matches the first of a composite step in userSteps
            hypMatch = filter(lambda x: "-" in x,filter(lambda x: step == x.split("-")[0],userSteps))
            if not len(hypMatch) == 0 :
                hypsteps    = expandHyphens(hypMatch[0])
                stepToWrite = ",".join(hypsteps)
                befStep     = hypsteps[0]
                aftStep     = hypsteps[-1]
                oneShotProf = True

            writeStepHead(simcandles,acandle,stepToWrite)

            #Set the output file name for Pile up and for regular case:
            if '--pileup' in cmsDriverOptions:
                outfile = stepToWrite + "_PILEUP"
            else:
                outfile = stepToWrite
                
            OutputFile = setOutputFileOption(acandle,outfile)
            if fstROOTfile:
                fstROOTfileStr = OutputFile
                fstROOTfile = False
            OutputFileOption = "--fileout=" + OutputFile

            for prof in Profile:
                #First prepare the cmsDriver.py command
                
                #Special case of EventEdmSize profiling 
                if 'EdmSize' in prof:
                    EdmFile = "%s_%s.root" % (FileName[acandle],outfile) #stepToWrite) #Bug in the filename for EdmSize for PileUp corrected.
                    #EventEdmSize needs a pre-requisite step that just produces the root file if one decided to run with only EdmSize profiling!
                    if prof == Profile[0] and not os.path.exists("./" + EdmFile):
                        # insert command to generate required state ( need to run one more step
                        # so that EDM can actually check the size of the root file
                        # If the first step to be profiled is something later on in the steps such
                        # as HLT then writePrerequisteSteps() should have got you to the step prior to
                        # HLT, therefore the only thing left to run to profile EDMSIZE is HLT itself

                        InputFileOption = "--filein file:" + previousOutputFile
                        if rawreg.search(step) and bypasshlt:
                            InputFileOption = "--filein file:" + prevPrevOutputFile
                        if previousOutputFile == "":
                            InputFileOption = setInputFile(steps,stepToWrite,acandle,stepIndex,pileup=pileup,bypasshlt=bypasshlt)
                        Command = ("%s %s -n %s --step=%s %s %s --customise=%s %s"
                                           % (cmsDriver,
                                              KeywordToCfi[acandle],
                                              NumberOfEvents,
                                              stepToWrite,
                                              InputFileOption,
                                              OutputFileOption,
                                              CustomisePythonFragment,
                                              cmsDriverOptions) )
                        simcandles.write( "%s @@@ None @@@ None\n" % (Command))

                    Command = EdmFile
                #all other profiles:
                else:
                    InputFileOption = "--filein file:" + previousOutputFile
                    if rawreg.search(step) and bypasshlt:
                        InputFileOption = "--filein file:" + prevPrevOutputFile

                    if previousOutputFile == "":
                        InputFileOption = setInputFile(steps,befStep,acandle,stepIndex,pileup,bypasshlt)

                    #if '--pileup' in cmsDriverOptions:
                    #    stepToWrite = pileupStep
                    if '--pileup' in cmsDriverOptions and ( stepToWrite=='GEN,SIM' or stepToWrite=='SIM'):
                        Command = ("%s %s -n %s --step=%s %s %s --customise=%s %s" % (
                               cmsDriver,
                               KeywordToCfi[acandle],
                               NumberOfEvents,
                               stepToWrite,
                               InputFileOption,
                               OutputFileOption,
                               CustomiseFragment['GEN,SIM'],#Done by hand to avoid silly use of MixinModule.py for pre-digi individual steps
                               cmsDriverOptions[:cmsDriverOptions.index('--pileup')]
                           ))
                    else:
                        Command = ("%s %s -n %s --step=%s %s %s --customise=%s %s" % (
                               cmsDriver,
                               KeywordToCfi[acandle],
                               NumberOfEvents,
                               stepToWrite,
                               InputFileOption,
                               OutputFileOption,
                               CustomisePythonFragment,
                               cmsDriverOptions
                           ))
                #Now the cmsDriver command is ready in Command, we just edit the rest of the line and write it to the file!
                if _noprof:
                    simcandles.write("%s @@@ None @@@ None\n" % Command)
                else:
                    stepLabel=stepToWrite
                    if '--pileup' in cmsDriverOptions and not "_PILEUP" in stepToWrite:
                        stepLabel = stepToWrite+"_PILEUP"
                    simcandles.write("%s @@@ %s @@@ %s_%s_%s\n" % (Command,
                                                                   Profiler[prof],
                                                                   FileName[acandle],
                                                                   stepLabel,
                                                                   prof))

                if debug:
                    print InputFileOption, step, 'GEN,SIM' in step, 'HLT' in steps[stepIndex - 1], steps
                    print "cmsDriveroptions : " + cmsDriverOption
            prevPrevOutputFile = previousOutputFile          
            previousOutputFile = OutputFile
        else:
            unprofiledSteps.append(step)
            isNextStepForProfile = False # Just an initialization for scoping. don't worry about it being false
            
            try:
                isNextStepForProfile = steps[stepIndex + 1] in userSteps or reduce(lambda x,y : x or y,map(lambda z: steps[ stepIndex + 1 ] == z.split("-")[0],userSteps))
            except IndexError:
                # This loop should have terminated if x + 1 is out of bounds!
                print "Error: Something is wrong we shouldn't have come this far"
                break

            if isNextStepForProfile:
                writeUnprofiledSteps(simcandles,CustomisePythonFragment,cmsDriverOptions,unprofiledSteps,acandle,NumberOfEvents,stepIndex,bypasshlt)
                unprofiledSteps = []
                
        #Dangerous index handling when looping over index x!        
        if oneShotProf:
            stepIndex += len(hypsteps)            
        else:
            stepIndex +=1
    return fstROOTfileStr

def preparePileUpCommand(rootinput,thecandle,NumberOfEvents,cmsDriverOptions):

    InputFileOption  = "--filein file:%s" % (rootinput)
    OutputFileOption = "--fileout=%s_DIGI_PILEUP.root"  % (FileName[thecandle] )

    return (
        "%s %s -n %s --step=DIGI %s %s --pileup=%s --customise=Validation/Performance/MixingModule.py %s" %
        (cmsDriver,
         KeywordToCfi[thecandle],
         NumberOfEvents,
         InputFileOption,
         OutputFileOption,
         cmsDriverPileUp,
         cmsDriverOptions))

def writeCommandsToReport(simcandles,Candle,Profile,debug,NumberOfEvents,cmsDriverOptions,steps,bypasshlt):

    OutputStep = ''

    # This option will not be used for now since the event content names keep changing, it can be
    # edited by hand pre-release by pre-release if wanted (Still moved all to FEVTDEBUGHLT for now, except RECO, left alone).
    #
    #EventContent = {'GEN,SIM': 'FEVTDEBUGHLT', 'DIGI': 'FEVTDEBUGHLT'}

    digiPUrootIN = ""
    #Using RunDigiPileUp dictionary in cmsPerfCommons.py to not have DIGI PILEUP hardcoded, but configurable
    #in the file above.
    #qcdStr = Candles[6]
    DigiPileUpWillRun = False
    for acandle in Candle:
        print '*Candle ' + acandle
        
        ##################
        # If the first profiling we run is EdmSize we need to create the root file first
        #

        #Here all regular (non pileup candles) are processed with all the default steps:
        fstoutfile = writeCommands(simcandles,
                                   Profile,
                                   acandle,
                                   steps,
                                   NumberOfEvents,
                                   cmsDriverOptions,
                                   bypasshlt)
        
        ##FIXME with the new way to run PU!
        #Here starts the special treatment in case of pileup (should change variable names from qcd-related to pileup as it logically is):
        #if acandle in DigiPileUpCandles:
        #    digiPUrootIN = fstoutfile
        #    DigiPileUpWillRun = True

    # Add the extra "step" DIGI with PILE UP only for QCD_80_120:
    # After digi pileup steps:
    # Freeze this for now since we will only run by default the GEN-SIM,DIGI and DIGI pileup steps

    #In case pile up is requested, check that the NumberOfEvents is more than the minimum number necessary for the MixingModule not to get stuck forever: 
        if DigiPileUpWillRun and MIN_REQ_TS_EVENTS <= NumberOfEvents: 
            #This piece of code seems unnecessary!
            #thecandle = getFstOccur(qcdStr, Candle)

            # First run the DIGI with PILEUP (using the MixingModule.py)
            # Hardcode stuff for this special step

            writeStepHead(simcandles,acandle,"DIGI PILE-UP")
            #print "Here's Profile list:%s"%Profile
            for prof in Profile:
                if 'EdmSize' in prof:
                    Command = "%s_DIGI_PILEUP.root " % (FileName[acandle])
                else:
                    Command = preparePileUpCommand(digiPUrootIN,acandle,NumberOfEvents,cmsDriverOptions)
                if _noprof:
                    simcandles.write("%s @@@ None @@@ None \n" % Command)
                else:
                    simcandles.write('%s @@@ %s @@@ %s_DIGI_PILEUP_%s\n' % (Command, Profiler[prof], FileName[acandle], prof))

            # Very messy solution for now:
            # Setting the stepIndex variable to 2, i.e. RECO step
            
        #The following 2 lines seem remnants of James' translation from Perl to Python...    
        #FileIn = {}
        #FileIn['RECO'] = '--filein file:'

        #We should consider a special function to handle the pileup case instead of making the writeCommands function quite hard to read...
        
            writeCommands(simcandles,
                          Profile,
                          acandle,
                          map(lambda x: x + "_PILEUP",AfterPileUpSteps),
                          NumberOfEvents,
                          cmsDriverOptions,
                          bypasshlt,
                          0, # start at step index 2, RECO Step
                          True) #this is the flag that says do PILEUP!
        
        elif NumberOfEvents < MIN_REQ_TS_EVENTS:
            print " WARNING: PileUp steps will not be run because the number of events is less than %s" % MIN_REQ_TS_EVENTS
        

def main(argv=sys.argv):

    #####################
    # Options Parser 
    #

    (options, args) = optionparse()

    #####################
    # Set up arguments and option handling
    #

    (NumberOfEvents, ProfileCode, cmsDriverOptions, steps, Candle, bypasshlt ) = setupProgramParameters(options,args)

    ######################
    # Initialize a few variables
    #

    (CMSSW_BASE, CMSSW_RELEASE_BASE, CMSSW_VERSION ) = init_vars()

    ##################
    # Ok everything is ready now we need to create the input file for the relvalreport script 
    #

    simcandles = writeReportFileHeader(CMSSW_VERSION,CMSSW_RELEASE_BASE,CMSSW_BASE)

    ##################
    # Based on the profile code create the array of profiles to run:
    #

    Profile = getProfileArray(ProfileCode)

    ##################
    # Write the commands for the report to the file
    #

    writeCommandsToReport(simcandles,Candle,Profile,debug,NumberOfEvents,cmsDriverOptions,steps,bypasshlt)
                
    simcandles.close()

if __name__ == "__main__":
    main()
