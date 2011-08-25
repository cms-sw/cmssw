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
from cmsPerfCommons import Candles, CandDesc, FileName, KeywordToCfi, CustomiseFragment, CandFname, EventContents
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
    'IgProfperf @@@ reuse'    : 'IgProf_perf.PERF_TICKS',
    'IgProfMemTotal'          : 'IgProf_mem.MEM_TOTAL',
    'IgProfMemTotal @@@ reuse': 'IgProf_mem.MEM_TOTAL',
    'IgProfMemLive'           : 'IgProf_mem.MEM_LIVE',
    'IgProfMemLive @@@ reuse' : 'IgProf_mem.MEM_LIVE',
    'IgProfMemMax'            : 'IgProf_mem.MEM_MAX',
    'IgProfMemMax @@@ reuse'  : 'IgProf_mem.MEM_MAX',
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
    '''Checks user steps <steps> for order against steps defined in AllSteps list.
    If they are not, prints a warning and exits.
    No return value.'''
    
    #print steps
    #if ":" in steps:
    #    steps.split(
    idx = -2
    lstidx = -2
    for step in steps:
        astep = step
        split = []
        #Added special workflow for GEN-FASTSIM, bypass the check of step order:
        if step=="GEN,FASTSIM":
            pass
        else:
            if "-" in step:
                split = astep.split("-")
                astep = split[0].split(":")[0]
             #   print "This is astep:%s"%astep
            #idx =AllSteps.index("GEN,SIM")
            #print idx
            #print astep
            idx = AllSteps.index(astep.split(":")[0])            
            if not ( idx == -2 ):
                if lstidx > idx:
                    print "ERROR: Your user defined steps are not in a valid order"
                    sys.exit()
                lstidx = idx    
            if "-" in step:
                lstidx = AllSteps.index(split[1].split(":")[0])


def getSteps(userSteps):
    '''Analyzes the steps in <userSteps> to translate the cmsRelvalreportInput.py --step option into the cmsDriver.py one. Expanding hyphens for combined steps.
    Handle the exceptional cases of GEN-, GEN-SIM and GEN-FASTSIM (GEN is never profiled by itself).
    Returns a list of steps <steps>.'''
    #Add handling of ":" in steps
    #print userSteps
    # Then split the user steps into "steps"
    gsreg = re.compile('GEN-SIM')
    greg = re.compile('GEN') #Add a second hack (due to the first) to handle the step 1 case GEN-HLT
    gfsreg = re.compile('GEN-FASTSIM') #Add a third hack to handle the case GEN-FASTSIM workflow
    StepsTokens = userSteps.split(",")
    steps = [] 
    for astep in StepsTokens:

        # Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO)
        # from using the "-" to using the "," to match cmsDriver.py convention

        if gsreg.search(astep):
            astep = gsreg.sub(r"GEN,SIM", astep)
        elif gfsreg.search(astep):
            astep = gfsreg.sub(r"GEN,FASTSIM", astep) #could add protection for user screw-ups
        elif greg.search(astep):
            astep = greg.sub(r"GEN,SIM", astep)
            
            
        #print astep
        # Finally collect all the steps into the steps list:

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

    Examples (could be obsolete!):  
       Perform Timing Report profiling for all candles, all steps and 10 events per cfg.
        ./%s 10 AllCandles 1 \"--conditions FakeConditions --eventcontent FEVTDEBUGHLT\"
       Perform Timing Report, Time Report and SimpleMemoryCheck profiling for HiggsZZ4LM200, all steps and 50 events.
        ./%s 50 \"HiggsZZ4LM200\" 012 \"--conditions FakeConditions --eventcontent FEVTDEBUGHLT\"
       Perform IgProfPerf and IgProfMemTotal profiling for TTbar, all steps and 100 events.
        ./%s 100 \"TTbar\" 45 --cmsdriver=\"--conditions FakeConditions --eventcontent RAWRECO\"
       Perform ValgrindFCE ValgrindMemCheck profiling for Minimum bias and 100 events. Only on GEN,SIM and DIGI steps.
        ./%s 100 \"MinBias\" 89 --cmsdriver=\"--conditions FakeConditions --eventcontent RAWSIM\" \"--usersteps=GEN-SIM,DIGI"""
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
    parser.add_option(
        '-i',
        '--filein',
        type='string',
        dest='userInputFile',
        help='Eventual input file (for now to be only used with HLT and RAW2DIGI-RECO workflows)',
        default='',
        metavar='<INPUTFILE>',
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
        #if not (len(hypsteps) == 2):
        #    print "ERROR: Could not parse usersteps. You have too many hypens between commas"
        #    sys.exit()
        #el
        if not reduce(lambda x,y,: x and y,map(lambda x: x.split(":")[0] in AllSteps, hypsteps)):
        #(hypsteps[0].split(":")[0] in AllSteps and hypsteps[1].split(":") in AllSteps):
            print "ERROR: One of the steps you defined is invalid"
            sys.exit()
        else:
            if (hypsteps[0] == hypsteps[1]):
                print "WARNING: You should not add a hypenated step that as the same source and destination step, ignoring"
                newsteps.append(hypsteps[0])
            else:
                newsteps.append(hypsteps[0])
                
                srt = AllSteps.index(hypsteps[0].split(":")[0]) + 1
                for n in range(srt,AllSteps.index(hypsteps[-1].split(":")[0])):
                    astep = AllSteps[n]
                    if astep in ( hypsteps[i].split(":")[0] for i in range(1,len(hypsteps))):
                        #break
                        astep=filter(lambda x: astep == x.split(":")[0],hypsteps)[0]
                        if astep==hypsteps[-1].split(":")[0]:
                            break
                    newsteps.append(astep)
                newsteps.append(hypsteps[-1])
    else:
        if not (step.split(":")[0] in AllSteps):
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

    if WhichCandles.lower() == 'allcandles':
        Candle = Candles
        print 'ALL standard simulation candles will be PROCESSED:'
    else:
        Candle = [WhichCandles]
        print 'Candle %s will be PROCESSED' % Candle[0]

    # For now the two steps are built in, this can be added as an argument later
    # Added the argument option so now this will only be defined if it was not defined already:

    if not steps:
        print 'The default steps will be run:'
        steps = DEF_STEPS
    else:
        print "You defined your own steps to run:"

    for astep in steps:
        print astep

    return (NumberOfEvents, ProfileCode, cmsDriverOptions, steps, Candle, options.bypasshlt,options.userInputFile)

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
        #'IgProfMemMax', #Should add this here... but instead of changing the profile code numbers (really should, using options instead of numbers...) I am hacking the IgProfMemLive one: when this one is selected I will also include the IgProfMemMax profile (and analysis is analyse is there)
        'IgProfMemAnalyse',
        'valgrind',
        'memcheck_valgrind',
        'None',
        ]

    if _noprof:
        Profile.append(AllowedProfile[-1])
    else:
        #FIXME:
        #Horrible code!
        for i in range(10):
            if str(i) in ProfileCode:
                firstCase = (i == 0 and (str(1) in ProfileCode or str(2) in ProfileCode)) \
                            or (i == 1 and str(2) in ProfileCode)
                secCase   = (i==4 and str(7) in ProfileCode) \
                            or (i == 5 and (str(6) in ProfileCode or str(7) in ProfileCode)) \
                            or (i == 6 and str(7) in ProfileCode)
                
                if firstCase or secCase:
                    Profile.append(AllowedProfile[i] + ' @@@ reuse')
                    #Here's the hack:
                    if i==6: #i.e. IgProfMemLive
                        Profile.append("IgProfMemMax @@@ reuse")
                else:
                    Profile.append(AllowedProfile[i])
                    if i==6: #i.e. IgProfMemLive
                        Profile.append("IgProfMemMax")

    return Profile

def writeStepHead(simcandles,acandle,step):
    simcandles.write('#%s\n' % FileName[acandle])
    simcandles.write('#Step %s\n' % step)
    #print step


def determineNewProfile(step,Profile,SavedProfile):
    if 'DIGI2RAW' in step:
        SavedProfile = Profile
        Profile = [ ]
    if 'HLT' in step:
        Profile = SavedProfile

    return (Profile, SavedProfile)

def pythonFragment(step,cmsdriverOptions):
    # Convenient CustomiseFragment dictionary to map the correct customise Python fragment for cmsDriver.py:
    #It is now living in cmsPerfCommons.py!
    
    if "--pileup" in cmsdriverOptions and not (step == "HLT" or step.startswith("RAW2DIGI")):
        return CustomiseFragment['DIGI-PILEUP']
    elif CustomiseFragment.has_key(step):
        return CustomiseFragment[step] 
    else:
        #This is a safe default in any case,
        #the maintenance of the customise python fragments goes into cmsPerfCommons.py
        return CustomiseFragment['DIGI']


def setInputFile(steps,step,acandle,stepIndex,pileup=False,bypasshlt=False):
    #print "StepIndex 2:%s"%stepIndex
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
        if 'GEN,FASTSIM' in step: # there is no input file for GEN,FASTSIM!
            InputFileOption = ''
        elif   'HLT' in steps[stepIndex - 1] and bypasshlt:

            # Special hand skipping of HLT since it is not stable enough, so it will not prevent
            # RAW2DIGI,RECO from running

            InputFileOption = "--filein file:%s_%s"  % ( FileName[acandle],steps[stepIndex - 2] )

    if not InputFileOption == "" :
        InputFileOption += ".root "

    return InputFileOption

def writeUnprofiledSteps(simcandles,CustomisePythonFragment,cmsDriverOptions,unprofiledSteps,previousOutputFile,acandle,NumberOfEvents, stepIndex, pileup,bypasshlt):
    # reduce(lambda x,y : x + "," + "y",unprofiledSteps)
    #stepsStr = ",".join(unprofiledSteps)

    
    #print "unprofiledSteps is %s"%unprofiledSteps
    #print "22acandle is %s"%acandle
    #Kludge in case -b option to skip HLT is used...
    if bypasshlt and unprofiledSteps[-1]=="HLT":
        stepsStr = ",".join(unprofiledSteps[:-1])
        OutputFile = "%s_%s.root" % ( FileName[acandle],unprofiledSteps[-2])
    else:
        stepsStr = ",".join(unprofiledSteps)
        OutputFile = "%s_%s.root" % ( FileName[acandle],unprofiledSteps[-1])
    simcandles.write("\n#Run a %s step(s) that has not been selected for profiling but is needed to run the next step to be profiled\n" % (stepsStr))
    OutputFileOption = "--fileout=%s" % OutputFile
    #Bug here: should take into account the flag --bypass-hlt instead of assuming hlt should be bypassed
    #This affects the Step1/Step2 running since Step1 will produce an HLT.root file and Step2 should start from there!
    #Adding the argument bypasshlt to the calls...
    #PreviousInputFile=AllSteps[AllSteps.index(unprofiledSteps[0])-1]
    #print "StepIndex 1:%s"%stepIndex
    #Correcting a bug: when unprofiled intermediate steps are present it would skip 1 step...
    #stepIndexAdjust=stepIndex - 2
    #if stepIndexAdjust < 0: #To avoid issues with negative indeces
    #    stepIndexAdjust=0
    #InputFileOption = setInputFile(AllSteps,unprofiledSteps[0],acandle,stepIndexAdjust,bypasshlt=bypasshlt)
    #Use an argument to make it easier:
    InputFileOption = "--filein file:" + previousOutputFile
    if previousOutputFile =="":
        InputFileOption = setInputFile(AllSteps,unprofiledSteps[0],acandle,stepIndex,pileup=pileup,bypasshlt=bypasshlt)
    #Introduce an over-ride of cmsDriverOptions:
    #For the case of unprofiled steps, always run them with FEVTDEBUGHLT eventcontent
    #At the moment the only use case is when running step2 on its own...
    for eventcontent in EventContents:
        cmsDriverOptions=re.sub(eventcontent,'FEVTDEBUGHLT',cmsDriverOptions)
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

def writePrerequisteSteps(simcandles,steps,acandle,NumberOfEvents,cmsDriverOptions,pileup,bypasshlt):
    fstIdx = -1
    #Hack for GEN-FASTSIM workflow:
    #if steps=="GEN,FASTSIM":   
    #if "-" in steps[0]:
    fstIdx = AllSteps.index(steps[0].split("-")[0].split(":")[0])
    #else:
    #    fstIdx = AllSteps.index(steps[0])
    CustomisePythonFragment = pythonFragment("GEN,SIM",cmsDriverOptions)
    previousOutputFile=""
    OutputFile = writeUnprofiledSteps(simcandles, CustomisePythonFragment, cmsDriverOptions,AllSteps[0:fstIdx],previousOutputFile,acandle,NumberOfEvents, 0,pileup,bypasshlt) 
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
                  pileup = False,
                  userInputFile=""):
    #print "a:%s,b:%s,c:%s"%(stepIndex,pileup,userInputFile)
    OutputStep = ""

    stopIndex = len(steps)
    start = 0

    userSteps = steps
    SavedProfile = []
    fstROOTfile = True
    fstROOTfileStr = ""

    #Handling the case of the first user step not being the first step (GEN,SIM):
    #print "Steps passed to writeCommands %s",steps
    #Special case for FASTSIM:
    if (steps[0] == "GEN,FASTSIM"):
        start=-1
        stopIndex=1
        steps = ["GEN,FASTSIM"]
    else:
        if not (steps[0] == AllSteps[0]) and (steps[0].split("-")[0] != "GEN,SIM"):
            #print "userInputFile: %s"%userInputFile
            if ("--pileup" in cmsDriverOptions) and (steps[0]=="HLT" or steps[0].startswith("RAW2DIGI")) :
                userInputFile = "../INPUT_PILEUP_EVENTS.root"
                stepIndex=AllSteps.index(steps[0].split("-")[0].split(":")[0])
                rootFileStr=""
                
            elif userInputFile == "":
                #Write the necessary line to run without profiling all the steps before the wanted ones in one shot:
                (stepIndex, rootFileStr) = writePrerequisteSteps(simcandles,steps,acandle,NumberOfEvents,cmsDriverOptions,pileup,bypasshlt)
            else: #To be implemented if we want to have input file capability beyond the HLT and RAW2DIGI-RECO workflows
                #if "-" in steps[0]:
                stepIndex=AllSteps.index(steps[0].split("-")[0].split(":")[0])
                    #print "Here!"
                #else:
                    #print "There!"
                #    stepIndex=AllSteps.index(steps[0])
                rootFileStr=""
            #Now take care of setting the indeces and input root file name right for the profiling part...
            if fstROOTfile:
                fstROOTfileStr = rootFileStr
                fstROOTfile = False
            start = -1
            #if "-" in steps[0]:
            start = AllSteps.index(steps[0].split("-")[0].split(":")[0])
            #else:
            #    start = AllSteps.index(steps[0])
            lst = - 1
            if "-" in steps[-1]:
                lst   = AllSteps.index(steps[-1].split("-")[1].split(":")[0]) #here we are assuming that - is always between two steps no GEN-SIM-DIGI, this is GEN-DIGI
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
                stopIndex = AllSteps.index(steps[-1].split("-")[1].split(":")[0]) + 1
            else:
                
                stopIndex = AllSteps.index(steps[-1].split(":")[0]) + 1
                #print "Last step is %s and stopIndex is %s"%(steps[-1],stopIndex)
        steps = AllSteps
            
    unprofiledSteps = []
    rawreg = re.compile("^RAW2DIGI")

#Horrible structure... to be rewritten sooner or later...
    
#   FOR step in steps

    prevPrevOutputFile = ""
    #Add if to introduce the possibility to run on user selected input file.
    if userInputFile!="":
        previousOutputFile=userInputFile
    else:
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
        if step in userSteps or reduce(lambda x,y : x or y,map(lambda x: step == x.split("-")[0].split(":")[0],userSteps)): 

            if "GEN,FASTSIM" in userSteps: # HLT/L1 menu workaround
                stepToWrite = stepToWrite + ",HLT:GRun"
            elif "HLT" in userSteps:
                stepToWrite = stepToWrite.replace("HLT", "HLT:GRun")
            
            #Checking now if the current step matches the first of a composite step in userSteps
            hypMatch = filter(lambda x: "-" in x,filter(lambda x: step == x.split("-")[0],userSteps))
            if not len(hypMatch) == 0 :
                hypsteps    = expandHyphens(hypMatch[0])
                #print hypsteps
                #print hypMatch[0]
                stepToWrite = ",".join(hypsteps)
                if "GEN,SIM-HLT" in userSteps: # HLT/L1 menu workaround
                    stepToWrite = stepToWrite.replace("HLT","HLT:GRun")
                befStep     = hypsteps[0]
                #Kludge to avoid running HLT in composite steps if the -b option is chosen
                if bypasshlt and hypsteps[-1]=='HLT':
                    aftStep     = hypsteps[-2]
                else:
                    aftStep     = hypsteps[-1]
                oneShotProf = True
            if filter(lambda x: stepToWrite==x.split(":")[0],userSteps):
                #print "Inside the filter if!"
                #print stepToWrite
                #Here we are assuming that the steps specified with : are not duplicate in the userSteps command:
                #i.e. GEN-DIGI:SequenceA,HLT:1E31 is OK
                #GEN-DIGI:SequenceA,HLT:SequenceB,HLT:1E31 IS NOT (OF COURSE!) 
                stepToWrite=filter(lambda x: stepToWrite==x.split(":")[0],userSteps)[0]
                #print stepToWrite

            writeStepHead(simcandles,acandle,stepToWrite)

            #Set the output file name for Pile up and for regular case:
            if "GEN,SIM-HLT" or "GEN,FASTSIM" or "HLT" in userSteps: # change to make root file output work (without colons)
                stepToWrite = stepToWrite.replace(":","=")

            if '--pileup' in cmsDriverOptions:
                outfile = stepToWrite + "_PILEUP"
            else:
                outfile = stepToWrite

            #print stepToWrite

            if "GEN,SIM-HLT" or "GEN,FASTSIM" or "HLT" in userSteps: # change it back
                stepToWrite = stepToWrite.replace("=",":")

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
                        #Kludge to bypass HLT output... here's a problem... since the unprofiled steps could contain HLT!
                        if rawreg.search(step) and bypasshlt and 'DIGI2RAW.root' in prevPrevOutputFile:
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
                    if rawreg.search(step) and bypasshlt and 'DIGI2RAW.root' in prevPrevOutputFile:
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
                    elif '--pileup' in cmsDriverOptions and (stepToWrite=='GEN,FASTSIM' or stepToWrite=='GEN,FASTSIM,HLT:GRun'): #Adding GEN,FASTIM case exception and use the CustomiseFragment['DIGI']
                                                                                                                                #+exception to the HLT:GRun case
                        Command = ("%s %s -n %s --step=%s %s %s --customise=%s %s" % (
                               cmsDriver,
                               KeywordToCfi[acandle],
                               NumberOfEvents,
                               stepToWrite,
                               InputFileOption,
                               OutputFileOption,
                               CustomiseFragment['DIGI'],#Done by hand to avoid silly use of MixinModule.py for pre-digi individual steps
                               cmsDriverOptions #For FASTSIM PU need the whole cmsDriverOptions! [:cmsDriverOptions.index('--pileup')] 
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
                    if 'valgrind' in prof:
                        #I think the if is safer even though all commands in this "else" (i.e. non EdmSize) should be OK.
                        if 'TimeMemoryInfo.py' in Command:
                            Command=Command.replace('--customise=Validation/Performance/TimeMemoryInfo.py', '')
                        #There is actually also the SIM case to take care of:
                        elif 'TimeMemoryG4Info.py' in Command:
                            Command=Command.replace('--customise=Validation/Performance/TimeMemoryG4Info.py', '')
                    stepLabel=stepToWrite
                    #print "stepLabel is %s"%stepLabel
                    if '--pileup' in cmsDriverOptions and not "_PILEUP" in stepToWrite:
                        stepLabel = stepToWrite+"_PILEUP"
                    #Add kludge here to make the IGPROF ANALYSE work:
                    #The idea is to:
                    #1-Check if ANALYSE is there everytime we're handling IGPROF profiles
                    #2-If it is there then write 2 lines instead of 1: add a @@@ reuse to the first line and a second line with ANALYSE @@@ the same metadata
                    #[3-Catch the case of ANALYSE by itself and raise an exception]
                    #4-Catch the case of ANALYSE proper, and do nothing in that case (already taken care by the second lines done after PERF_TICKS, MEM_TOTAL and MEM_LIVE.
                    #print "EEEEEEE %s"%prof
                    #We actually want the analyse to be used also for IgProf perf!
                    if 'IgProf' in prof: # and 'perf' not in prof:
                        #New IgProf naming scheme:
                        #Add a new naming scheme to simplify the handling in igprof-navigator:
                        #Candle___Step___Conditions__PileUpOption__IgProfProfile___Event.gz (and .sql3)

                        #Creating the IgProfProfile variable, to be either IgProfMem or IgProfPerf, the two types of profiling:
                        IgProfProfile="IgProf"
                        if "Mem" in prof:
                            if "reuse" in prof:
                                IgProfProfile="IgProfMem @@@ reuse"
                            else:
                                IgProfProfile="IgProfMem"
                        else:
                            if "reuse" in prof:
                                IgProfProfile="IgProfPerf @@@ reuse"
                            else:
                                IgProfProfile="IgProfPerf"

                        #Conditions and PileUpOption are extracted from the cmsDriverOptions:
                        Conditions="UNKNOWN_CONDITIONS"
                        PileUp="NOPILEUP"
                        EventContent="UNKNOWN_EVENTCONTENT"
                        tokens=cmsDriverOptions.split("--")
                        for token in tokens:
                            keywords=token.split(" ")
                            if "=" in keywords[0]:
                                keywords=keywords[0].split("=")
                            if "conditions" in keywords[0]:
                                # check if we are using the autoCond style of flexible conditions
                                # if so, expand the condition here so that the file names contain the real conditions
                                if "auto:" in keywords[1]:
                                    from Configuration.AlCa.autoCond import autoCond
                                    fileConditionsOption = autoCond[ keywords[1].split(':')[1] ]
                                    Conditions = autoCond[ keywords[1].split(':')[1] ].split("::")[0] 
                                else:
                                    # old style, hardcoded, conditions ...
                                    # Complicated expression, just to get rid of FrontierConditions_GlobalTag,
                                    # and ::All at the end:
                                    # Added if to handle new --conditions convention for cmsDriver.py (no more
                                    # FrontierConditions... in front of it)
                                    if "," in keywords[1]: 
                                        Conditions=keywords[1].split(",")[1].split("::")[0] #"backward" compatibility...
                                    else:
                                        Conditions=keywords[1].split("::")[0] 
                            elif "pileup" in keywords[0]:
                                PileUp=keywords[1]
                            elif "eventcontent" in keywords[0]:
                                EventContent=keywords[1]

                        #so here's our new MetaName:
                        MetaName=acandle+"___"+stepToWrite+"___"+PileUp+"___"+Conditions+"___"+EventContent+"___"+IgProfProfile
                        
                        if 'Analyse' not in prof and (lambda x: 'Analyse' in x,Profile):
                                
                            simcandles.write("%s @@@ %s @@@ %s\n" % (Command,
                                                                           Profiler[prof],
                                                                           MetaName))
                                                                           #FileName[acandle],
                                                                           #stepLabel,
                                                                           #prof))
                            analyse_command = ("%s @@@ %s @@@ %s\n" % (Command,
                                                                           Profiler[prof].split(".")[0]+'.ANALYSE.'+Profiler[prof].split(".")[1], #Add the IgProf counter in here (e.g. IgProfMem.ANALYSE.MEM_TOTAL)
                                                                           MetaName))
                                                                           #FileName[acandle],
                                                                           #stepLabel,
                                                                           #prof))

                            #for the relval case, GEN-SIM,HLT get rid of the last " @@@ reuse":
                            if Profiler[prof] == 'IgProf_mem.MEM_MAX' or Profiler[prof] == 'IgProf_perf.PERF_TICKS':
                                analyse_command = analyse_command.replace("@@@ reuse", "")

                            simcandles.write(analyse_command)

                        elif 'Analyse' in prof:
                            pass
                    else:    
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
                isNextStepForProfile = steps[stepIndex + 1] in userSteps or reduce(lambda x,y : x or y,map(lambda z: steps[ stepIndex + 1 ] == z.split("-")[0].split(":")[0],userSteps))
                #print "YES found Step %s when looking with standard step %s"%(steps[stepIndex],steps[stepIndex])
            except IndexError:
                # This loop should have terminated if x + 1 is out of bounds!
                print "Error: Something is wrong we shouldn't have come this far"
                break

            if isNextStepForProfile:
                #Minimum intervention solution is to fix it here: I think we need to pass the before and after steps...
                OutputFile=writeUnprofiledSteps(simcandles,CustomisePythonFragment,cmsDriverOptions,unprofiledSteps,previousOutputFile,acandle,NumberOfEvents,stepIndex,pileup,bypasshlt)
                unprofiledSteps = []
                prevPrevOutputFile = previousOutputFile          
                previousOutputFile = OutputFile
        #Dangerous index handling when looping over index x!        
        if oneShotProf:
            stepIndex += len(hypsteps)            
        else:
            stepIndex +=1
    return fstROOTfileStr

def writeCommandsToReport(simcandles,Candle,Profile,debug,NumberOfEvents,cmsDriverOptions,steps,bypasshlt,userInputFile):

    OutputStep = ''

    # This option will not be used for now since the event content names keep changing, it can be
    # edited by hand pre-release by pre-release if wanted (Still moved all to FEVTDEBUGHLT for now, except RECO, left alone).
    #
    #EventContent = {'GEN,SIM': 'FEVTDEBUGHLT', 'DIGI': 'FEVTDEBUGHLT'}

    for acandle in Candle:
        print '*Candle ' + acandle
        
        ##################
        # If the first profiling we run is EdmSize we need to create the root file first
        #

        #Here all candles are processed with all the same command, and in the pileup case they will have the pileup settings set correctly already:
        #print "1-userInputFile: %s"%userInputFile
        
        fstoutfile = writeCommands(simcandles,
                                   Profile,
                                   acandle,
                                   steps,
                                   NumberOfEvents,
                                   cmsDriverOptions,
                                   bypasshlt,
                                   0,
                                   False,
                                   userInputFile)
        

def main(argv=sys.argv):

    #####################
    # Options Parser 
    #

    (options, args) = optionparse()

    #####################
    # Set up arguments and option handling
    #

    (NumberOfEvents, ProfileCode, cmsDriverOptions, steps, Candle, bypasshlt, userInputFile ) = setupProgramParameters(options,args)
    #print setupProgramParameters(options,args)

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

    writeCommandsToReport(simcandles,Candle,Profile,debug,NumberOfEvents,cmsDriverOptions,steps,bypasshlt,userInputFile)

    print "Written out cmsRelvalreport.py input file at:\n%s"%os.path.abspath('./'+simcandles.name)
                
    simcandles.close()

if __name__ == "__main__":
    main()
