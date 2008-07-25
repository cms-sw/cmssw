#!/usr/bin/python
# -*- coding: utf-8 -*-
# GBenelli Dec 21 and JNicolson July 23
# This script is designed to run on a local directory
# after the user has created a local CMSSW release,
# initialized its environment variables by executing
# in the release /src directory:
# eval `scramv1 runtime -csh`
# project CMSSW
# The script will create a SimulationCandles.txt ASCII
# file, input to cmsRelvalreport.py, to launch the
# standard simulation performance suite.

# Input arguments are three:
# 1-Number of events to put in the cfg files
# 2-Name of the candle(s) to process (either AllCandles, or NameOfTheCandle)
# 3-Profiles to run (with code below)
# E.g.: ./cmsRelValReportInput.py 50 AllCandles 012

###############
# Import modules and define functions
#

import sys
import os
import re
import optparse as opt #import OptionParser # import OptionParser

steps = []
cmsDriverOptions = ""

def getFstOccur(str, list):
    for elem in list:
        if str == elem:
            return elem

def getSteps(userSteps, steps):

    # Then split the user steps into "steps"

    StepsTokens = userSteps.split(r",")
    for astep in StepsTokens:

        # Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO)
        # from using the "-" to using the "," to match cmsDriver.py convention

        if hypreg.search(astep):
            astep = hypreg.sub(r",", astep)

        # print astep
        # Finally collect all the steps into the @Steps array:

        steps.append(astep)
        
################
# Global variables
#

THIS_PROG_NAME = os.path.basename(sys.argv[0])
cmsDriver = 'cmsDriver.py'                    #cmsDriver.py path
hypreg = re.compile('-')

#####################
# Options Parser
#

parser = opt.OptionParser(usage=("""%s NUM_EVENTS_PER_CFG CANDLES PROFILE [--conditions=cmsDriverOptions] [--usersteps=processingStepsOption]

:

 NUM_EVENTS_PER_CFG - The number of events per config file

 CANDLES - The simulation type to perform profiling on
   AllCandles
   \"HZZLLLL\"       Higgs boson
   \"MINBIAS\"       Minimum Bias
   \"E -e 1000\"     
   \"MU- -e pt10\"   Muon
   \"PI- -e 1000\"   Pion
   \"TTBAR\"         TTbar
   \"QCD -e 80_120\" Quantum Chromodynamics

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

   \"--conditions FakeConditions\" - Set conditions

   EXAMPLES:
   ========
   
   Perform Timing Report profiling for all candles and 10 events per cfg.
    ./%s 10 AllCandles 1
   Perform Timing Report, Time Report and SimpleMemoryCheck profiling for Higgs boson and 50 events.
    ./%s 50 \"HZZLLLL\" 012
   Perform IgProfPerf and IgProfMemTotal profiling for TTbar and 100 events.
    ./%s 100 \"TTBAR\" 45 \"--conditions FakeConditions\"
   Perform ValgrindFCE ValgrindMemCheck profiling for Minimum bias and 100 events. Only on gensim and digi steps.
    ./%s 100 \"MINBIAS\" 89 \"--conditions FakeConditions\" \"--usersteps=GEN-SIM,DIGI

""" % ( THIS_PROG_NAME, THIS_PROG_NAME,THIS_PROG_NAME,THIS_PROG_NAME,THIS_PROG_NAME))) #% ([THIS_PROG_NAME for x in range(5)])), version="ads 1.0") #
devel  = opt.OptionGroup(parser, "Developer Options",
                                 "Caution: use these options at your own risk."
                                 "It is believed that some of them bite.\n")
#parser.set_defaults(debug=False)

parser.add_option(
    '-u',
    '--usersteps',
    type='string',
    dest='userSteps',
    help='Which steps to run',
    metavar='STEP(S)',
    )
parser.add_option(
    '-c',
    '--conditions',
    type='string',
    dest='cmsDriverOptions',
    help='Option for cmsDriver.py can be specified as a string to be added to all cmsDriver.py commands',
    metavar='CONDITION',
    )
devel.add_option(
    '-d',
    '--debug',
    action="store_true",
    dest='debug',
    help='Show debug output',
    metavar='DEBUG',
    )

parser.set_defaults(debug=False)
parser.add_option_group(devel)

(options, args) = parser.parse_args()

debug = options.debug
numofargs = len(args) 


if not numofargs == 3:
    parser.error("There are not enough arguments specified to run this program."
                 "Please use --help to determine the correct args")
    sys.exit()

#####################
# Set up arguments and option handling
#

NumberOfEvents = str(args[0])  # first arg
WhichCandles   = str(args[1])  # second arg
ProfileCode    = str(args[2])  # third arg

if options.userSteps:

    userSteps = options.userSteps
    getSteps(userSteps, steps)

if options.cmsDriverOptions:
    
    cmsDriverOptions = options.cmsDriverOptions
    cmsDriverOptions = "--conditions " + cmsDriverOptions
    print 'Using user-specified cmsDriver.py options: ' + cmsDriverOptions

if WhichCandles == 'AllCandles':
    Candle = (
        'HZZLLLL',
        'MINBIAS',
        'E -e 1000',
        'MU- -e pt10',
        'PI- -e 1000',
        'TTBAR',
        'QCD -e 80_120',
        )
    print 'ALL standard simulation candles will be PROCESSED:'
else:
    Candle = [WhichCandles]
    print 'ONLY %s will be PROCESSED' % Candle[0]

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

# Adding a check for a local version of the packages

PyRelValPkg = 'CMSSW_BASE' + '/src/Configuration/PyReleaseValidation'
if os.path.exists(PyRelValPkg):
    BASE_PYRELVAL = PyRelValPkg
    print '**[cmsSimPyRelVal.pl]Using LOCAL version of Configuration/PyReleaseValidation instead of the RELEASE version**'
else:
    BASE_PYRELVAL = CMSSW_RELEASE_BASE + '/src/Configuration/PyReleaseValidation'

######################
# Initialize a few variables
#

# Convenient hash to map the correct Simulation Python fragment:

CustomiseFragment = {
     'GEN,SIM': 'Configuration/PyReleaseValidation/SimulationG4.py',
     'DIGI': 'Configuration/PyReleaseValidation/Simulation.py'}

# This option will not be used for now since the event content names keep changing, it can be
# edited by hand pre-release by pre-release if wanted (Still moved all to FEVTDEBUGHLT for now, except RECO, left alone).

EventContent = {'GEN,SIM': 'FEVTDEBUGHLT', 'DIGI': 'FEVTDEBUGHLT'}

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
Profile = []

# Need a little hash to match the candle with the ROOT name used by cmsDriver.py.

FileName = {
    'HZZLLLL': 'HZZLLLL_190',
    'MINBIAS': 'MINBIAS_',
    'E -e 1000': 'E_1000',
    'MU- -e pt10': 'MU-_pt10',
    'PI- -e 1000': 'PI-_1000',
    'TTBAR': 'TTBAR_',
    'QCD -e 80_120': 'QCD_80_120',
    }

##################
# Ok everything is ready now we need to input file for the relvalreport script 
#

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

# For now the two steps are built in, this can be added as an argument later
# Added the argument option so now this will only be defined if it was not defined already:

if not steps:
    print 'The default steps will be run:'
    steps = ('GEN,SIM', 'DIGI')
else:

    print 'The default steps will be run:'

for astep in steps:
    print astep

# Based on the profile code create the array of profiles to run:

for i in range(10):
    if str(i) in ProfileCode:
        firstCase = i == 0 and (str(1) in ProfileCode or str(2) in ProfileCode) or i == 1 and str(2) in ProfileCode
        secCase   = i == 5 and (str(6) in ProfileCode or str(7) in ProfileCode) or i == 6 and str(7) in ProfileCode
        
        if firstCase or secCase:
            Profile.append(AllowedProfile[i] + ' @@@ reuse')
        else:
            Profile.append(AllowedProfile[i])

# Hash for the profiler to run

Profiler = {
    'TimingReport': 'Timing_Parser',
    'TimingReport @@@ reuse': 'Timing_Parser',
    'TimeReport': 'Timereport_Parser',
    'TimeReport @@@ reuse': 'Timereport_Parser',
    'SimpleMemReport': 'SimpleMem_Parser',
    'EdmSize': 'Edm_Size',
    'IgProfperf': 'IgProf_perf.PERF_TICKS',
    'IgProfMemTotal': 'IgProf_mem.MEM_TOTAL',
    'IgProfMemTotal @@@ reuse': 'IgProf_mem.MEM_TOTAL',
    'IgProfMemLive': 'IgProf_mem.MEM_LIVE',
    'IgProfMemLive @@@ reuse': 'IgProf_mem.MEM_LIVE',
    'IgProfMemAnalyse': 'IgProf_mem.ANALYSE',
    'valgrind': 'ValgrindFCE',
    'memcheck_valgrind': 'Memcheck_Valgrind',
    'None': 'None',
    }

# Hash to switch from keyword to .cfi use of cmsDriver.py:

KeywordToCfi = {
    'HZZLLLL': 'H200ZZ4L.cfi',
    'MINBIAS': 'MinBias.cfi',
    'E -e 1000': 'SingleElectronE1000.cfi',
    'MU- -e pt10': 'SingleMuPt10.cfi',
    'PI- -e 1000': 'SinglePiE1000.cfi',
    'TTBAR': 'TTbar.cfi',
    'QCD -e 80_120': 'QCD_Pt_80_120.cfi',
    }
OutputStep = ''
for acandle in Candle:
    print '*Candle ' + acandle
    stepIndex = 0
    for step in steps:
        simcandles.write('#%s\n' % FileName[acandle])
        simcandles.write('#Step %s\n' % step)
        print step
        if 'DIGI2RAW' in step:

        # print "DIGI2RAW"
        # print "$step;

            SavedProfile = Profile
            Profile = 'None'
        if 'HLT' in step:
            Profile = SavedProfile
        for prof in Profile:
            if 'EdmSize' in prof:

                # if ("GEN,SIM" in step): #Hack since we use SIM and not GEN,SIM extension (to facilitate DIGI)

                Command = "%s_%s.root " % (FileName[acandle],step)
            else:

                # step ="GEN\,SIM"
                # print "GEN\,SIM here"

                if not CustomiseFragment.has_key(step):

                # Temporary hack to have potentially added steps use the default Simulation.py fragment
                # This should change once each group customises its customise python fragments.

                    CustomisePythonFragment = CustomiseFragment['DIGI']
                else:
                    CustomisePythonFragment = CustomiseFragment[step]

                # Adding a fileout option too to avoid dependence on future convention changes in cmsDriver.py:

                OutputFileOption = "--fileout=%s_%s.root" % ( FileName[acandle],step)
                OutputStep = step

                # Use --filein (have to for L1, DIGI2RAW, HLT) to add robustness

                if 'GEN,SIM' in step:  # there is no input file for GEN,SIM!
                    InputFileOption = ''
                elif 'HLT' in steps[stepIndex - 1]:

                # Special hand skipping of HLT since it is not stable enough, so it will not prevent
                # RAW2DIGI,RECO from running

                    InputFileOption = "--filein file:%s_%s.root"  % ( FileName[acandle],steps[stepIndex - 2] )
                else:
                    InputFileOption = "--filein file:%s_%s.root " % ( FileName[acandle],steps[stepIndex - 1] )
                    
                if debug:
                    print InputFileOption, step, 'GEN,SIM' in step, 'HTL' in steps[stepIndex - 1], steps

                # Adding .cfi to use new method of using cmsDriver.py
                # $Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile
                # --customise=$CustomiseFragment{$step} ";
                # $Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=$step
                # $InputFileOption $OutputFileOption --eventcontent=$EventContent{$step}
                # --customise=$CustomiseFragment{$step} $cmsDriverOptions";

                if debug :
                    print "cmsDriveroptions : " + cmsDriverOptions

                Command = ("%s %s -n %s --step=%s %s %s --customise=%s %s" % (
                    cmsDriver,
                    KeywordToCfi[acandle],
                    NumberOfEvents,
                    step,
                    InputFileOption,
                    OutputFileOption,
                    CustomisePythonFragment,
                    cmsDriverOptions
                    ))
            simcandles.write("%s @@@ %s @@@ %s_%s_%s\n" % (Command,
                                                           Profiler[prof],
                                                           FileName[acandle],
                                                           OutputStep,
                                                           prof))
        stepIndex += 1

# Add the extra "step" DIGI with PILE UP only for QCD_80_120:
# After digi pileup steps:
# Freeze this for now since we will only run by default the GEN-SIM,DIGI and DIGI pileup steps

AfterPileUpSteps = []

qcdStr = 'QCD -e 80_120'
if qcdStr in Candle:
    thecandle = getFstOccur(qcdStr, Candle)

    # First run the DIGI with PILEUP (using the MixingModule.py)
    # Hardcode stuff for this special step

    simcandles.write('#' + FileName[qcdStr] + "\n")
    simcandles.write('#DIGI PILE-UP STEP\n')
    print 'DIGI PILEUP'
    for prof in Profile:
        if 'EdmSize' in prof:
            Command = FileName[thecandle] + '_DIGI_PILEUP.root '
        else:
            InputFileOption  = "--filein file:%s_GEN,SIM.root " % (FileName[thecandle])
            OutputFileOption = "--fileout=%s_DIGI_PILEUP.root"  % (FileName[thecandle] )

        # Adding .cfi to use new method of using cmsDriver.py
        # $Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile
        # --customise=$CustomiseFragment{$step} ";
        # $Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=DIGI $InputFileOption
        # $OutputFileOption --PU --eventcontent=FEVTSIMDIGI
        # --customise=Configuration/PyReleaseValidation/MixingModule.py $cmsDriverOptions";

            Command = (
                "%s %s -n %s --step=DIGI %s %s --PU --customise=Configuration/PyReleaseValidation/MixingModule.py %s"
                 % (
                cmsDriver,
                KeywordToCfi[thecandle],
                NumberOfEvents,
                InputFileOption,
                OutputFileOption,
                cmsDriverOptions,
                ))
        simcandles.write('%s @@@ %s @@@ %s_DIGI_PILEUP_%s\n' % (Command, Profiler[prof], FileName[thecandle], prof))

        # Very messy solution for now:
        # Setting the stepIndex variable to 2, i.e. RECO step

        stepIndex = 2
        FileIn = {}
        FileIn['RECO'] = '--filein file:'
        for step in AfterPileUpSteps:
            simcandles.write('#' + FileName[thecandle] + "\n")
            simcandles.write('#Step %s PILEUP\n' % step)
            print step + ' PILEUP'
            if 'DIGI2RAW' in step:
                SavedProfile = Profile
                Profile = 'None'
            if 'HLT' in step:
                Profile = SavedProfile

            for prof in Profile:
                if 'EdmSize' in prof:
                    Command = '%s_%s_PILEUP.root '% (FileName[thecandle], step)
                else:
                    if CustomiseFragment.has_key(step):

                    # Temporary hack to have potentially added steps use the default Simulation.py fragment
                    # This should change once each group customises its customise python fragments.

                        CustomisePythonFragment = CustomiseFragment['DIGI']
                    else:
                        CustomisePythonFragment = CustomiseFragment[step]
                    OutputStep       = step + '_PILEUP'
                    InputFileOption  = '%s_%s_PILEUP.root ' % (FileName[thecandle], Steps[stepIndex - 1])
                    OutputFileOption = '--fileout=%s_%s_PILEUP.root'% (FileName[candle], step)

                    # Adding .cfi to use new method of using cmsDriver.py
                    # $Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile
                    #--customise=$CustomiseFragment{$step} ";
                    # $Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=$step $InputFileOption
                    #$OutputFileOption --eventcontent=$EventContent{$step} --customise=$CustomiseFragment{$step} $cmsDriverOptions";

                    Command = (
                        '%s %s -n %s --step=%s %s %s --customise=%s %s'
                         % (
                        cmsDriver,
                        KeywordToCfi[Candle],
                        NumberOfEvents,
                        step,
                        InputFileOption,
                        OutputFileOption,
                        CustomisePythonFragment,
                        cmsDriverOptions,
                        ))
                simcandles.write('%s @@@ %s @@@ %s_%s_%s\n' % (Command, Profiler[prof], FileName[Candle], OutputStep, prof))
            stepIndex += 1

simcandles.close()
