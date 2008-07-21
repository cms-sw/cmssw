#!/usr/bin/python
#GBenelli Dec21 
#This script is designed to run on a local directory 
#after the user has created a local CMSSW release,
#initialized its environment variables by executing
#in the release /src directory:
#eval `scramv1 runtime -csh`
#project CMSSW
#The script will create a SimulationCandles.txt ASCII
#file, input to cmsRelvalreport.py, to launch the 
#standard simulation performance suite.

#Input arguments are three:
#1-Number of events to put in the cfg files
#2-Name of the candle(s) to process (either AllCandles, or NameOfTheCandle)
#3-Profiles to run (with code below)
#E.g.: ./cmsSimPyRelVal.py 50 AllCandles 012

#Get some environment variables to use
import sys
import os
import re

def getFstOccur(str,list):
    for elem in list:
        if (str == elem):
            return elem    

debug = 0

try:
    CMSSW_BASE=os.environ['CMSSW_BASE']
    CMSSW_RELEASE_BASE=os.environ['CMSSW_RELEASE_BASE']
    CMSSW_VERSION=os.environ['CMSSW_VERSION']
except KeyError:
    print "Error: An environment variable either CMSSW_{BASE, RELEASE_BASE or VERSION} is not available."
    print "       Please run eval `scramv1 runtime -csh` to set your environment variables"
    sys.exit()

#Adding a check for a local version of the packages
PyRelValPkg="CMSSW_BASE" + "/src/Configuration/PyReleaseValidation"
if os.path.exists(PyRelValPkg):
    BASE_PYRELVAL=PyRelValPkg
    print "**[cmsSimPyRelVal.pl]Using LOCAL version of Configuration/PyReleaseValidation instead of the RELEASE version**"
else:
    BASE_PYRELVAL=CMSSW_RELEASE_BASE + "/src/Configuration/PyReleaseValidation"

#Setting the path for the cmsDriver.py command:
cmsDriver= "cmsDriver.py"

numofargs = len (sys.argv) - 1

if not (( numofargs >= 3) and (numofargs <= 5)):
	print """Usage: cmsSimPyRelVal.pl NumberOfEventsPerCfgFile Candles Profile [cmsDriverOptions] [processingStepsOption]
Candles codes:
 AllCandles
 \"HZZLLLL\"
 \"MINBIAS\"
 \"E -e 1000\"
 \"MU- -e pt10\"
 \"PI- -e 1000\"
 \"TTBAR\"
 \"QCD -e 80_120\"
Profile codes (multiple codes can be used):
 0-TimingReport
 1-TimeReport
 2-SimpleMemoryCheck
 3-EdmSize
 4-IgProfPerf
 5-IgProfMemTotal
 6-IgProfMemLive
 7-IgProfAnalyse
 8-ValgrindFCE
 9-ValgrindMemCheck

Option for cmsDriver.py can be specified as a string to be added to all cmsDriver.py commands:
\"--conditions FakeConditions\"

Examples: 
./cmsSimulationCandles.pl 10 AllCandles 1 
OR 
./cmsSimulationCandles.pl 50 \"HZZLLLL\" 012\n
OR
./cmsSimulationCandles.pl 100 \"TTBAR\" 45 \"--conditions FakeConditions\"\n
OR
./cmsSimulationCandles.pl 100 \"MINBIAS\" 89 \"--conditions FakeConditions\" \"--usersteps=GEN-SIM,DIGI\"\n"""
	sys.exit()
        
NumberOfEvents= str(sys.argv[1]) #first arg
WhichCandles  = str(sys.argv[2]) #second arg
thirdarg      = str(sys.argv[3]) #third arg
if numofargs>=4:
    fortharg      = str(sys.argv[4]) #forth arg
if numofargs>=5:
    fiftharg      = str(sys.argv[5]) #fifth arg
usrreg = re.compile("--usersteps")
hypreg = re.compile("-")
steps = []
cmsDriverOptions = ""

if usrreg.match(thirdarg):
    userSteps=thirdarg
else:
    ProfileCode=thirdarg

if (numofargs==4):
    if usrreg.match(fortharg):
	#First split the option using the "=" to get actual user steps
	userStepsTokens = fortharg.split("=")
	userSteps= userStepsTokens[1]
	#print userSteps;
	#Then split the user steps into "steps"
        StepsTokens=userSteps.split(",")
        print "adsffdas " + StepsTokens
        #x= len StepTokens
        for astep in StepsTokens:
	    #Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO) 
	    #from using the "-" to using the "," to match cmsDriver.py convention
            if hypreg.search(astep):
		astep = hypreg.sub(r",",astep)
	    #print "$_";
	    #Finally collect all the steps into the @Steps array:
	    steps.append(astep)
    else:
	cmsDriverOptions=fortharg
	print "Using user-specified cmsDriver.py options: " + cmsDriverOptions
#Ugly cut and pastes for now, since we need to rewrite in python anyway...

if numofargs==5:
    if usrreg.match(fiftharg):
	#First split the option using the "=" to get actual user steps
	userStepsTokens=fiftharg.split("=")
	userSteps= userStepsTokens[1]
	#print userSteps
	#Then split the user steps into "steps"
        StepsTokens=userSteps.split(",")
        for astep in StepsTokens:
	    #Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO) 
	    #from using the "-" to using the "," to match cmsDriver.py convention
            if hypreg.search(astep):
		astep = hypreg.sub(r",",astep)
	    #print astep
	    #Finally collect all the steps into the @Steps array:
	    steps.append(astep)
	cmsDriverOptions=fortharg
	print "Using user-specified cmsDriver.py options: " + cmsDriverOptions
    elif usrreg.match(fortharg):
	#First split the option using the "=" to get actual user steps
	userStepsTokens=fortharg.split("=")
	userSteps= userStepsTokens[1]
	#print userSteps;
	#Then split the user steps into "steps"
	StepsTokens=userSteps.split(",")
        for astep in StepsTokens:
	    #Then transform the combined steps (GEN-SIM, RAW2DIGI-RECO) 
	    #from using the "-" to using the "," to match cmsDriver.py convention
            if hypreg.search(astep):
                astep = hypreg.sub(r",",astep)
	    #print astep
	    #Finally collect all the steps into the @Steps array:
	    steps.append(astep)
	cmsDriverOptions=fiftharg
	print "Using user-specified cmsDriver.py options: " + cmsDriverOptions

if WhichCandles == "AllCandles":
    Candle=("HZZLLLL",
	     "MINBIAS",
	     "E -e 1000",
	     "MU- -e pt10",
	     "PI- -e 1000",
	     "TTBAR",
	     "QCD -e 80_120"
	     )
    print "ALL standard simulation candles will be PROCESSED:"
else:
   
    Candle=[ WhichCandles ] 
    print "ONLY " + Candle[0] + " will be PROCESSED";

#Need a little hash to match the candle with the ROOT name used by cmsDriver.py.
FileName={
    "HZZLLLL":"HZZLLLL_190",
    "MINBIAS":"MINBIAS_",
    "E -e 1000":"E_1000",
    "MU- -e pt10":"MU-_pt10",
    "PI- -e 1000":"PI-_1000",
    "TTBAR":"TTBAR_",
    "QCD -e 80_120":"QCD_80_120"
    }
#Creating and opening the ASCII input file for the relvalreport script:
SimCandlesFile= "SimulationCandles"+"_"+CMSSW_VERSION +".txt"
try:
    simcandles = open(SimCandlesFile,"w")
except IOError:
    print "Couldn't open " + SimCandlesFile + " to save"
    
print >> simcandles, "#Candles file automatically generated by " + os.path.basename(sys.argv[0])  + " for " + CMSSW_VERSION
print >> simcandles, "#CMSSW Base  : " + CMSSW_BASE
print >> simcandles, "#Release Base: " + CMSSW_RELEASE_BASE
print >> simcandles, "#Version     : " + CMSSW_VERSION + "\n"

#For now the two steps are built in, this can be added as an argument later
#Added the argument option so now this will only be defined if it was not defined already:
if not steps:
    print "The default steps will be run:";
    steps=(
	   "GEN,SIM",
	   #"SIM",#To run SIM only need GEN done already!
	   "DIGI",
	   #Adding L1 step
	   #"L1",
	   #Adding DIGI2RAW step
	   #"DIGI2RAW",
	   #Adding HLT step
	   #"HLT",
	   #Adding RAW2DIGI step together with RECO
	   #"RAW2DIGI,RECO"
	   )
else:
    print "The default steps will be run:"
    
for astep in steps:
    print astep
    
#Convenient hash to map the correct Simulation Python fragment:
CustomiseFragment={
		    #Added the Configuration/PyReleaseValidation/ in front of the fragments names 
		    #Since the customise option was broken in 2_0_0_pre8, when the fragments were
		    #moved into Configuration/PyReleaseValidation/python from /data.
		    #unfortunately the full path Configuration/PyReleaseValidation/python/MyFragment.py
		    #does not seem to work.
    "GEN,SIM":"Configuration/PyReleaseValidation/SimulationG4.py",
    #"SIM":"SimulationG4.py", #To run SIM only need GEN done already!
    "DIGI":"Configuration/PyReleaseValidation/Simulation.py"
		    #,
		    #The following part should be edited to implement the wanted step and 
		    #define the appropriate customise fragment path
		    #Adding RAW2DIGI,RECO step
    #"RAW2DIGI,RECO":"Configuration/PyReleaseValidation/Simulation.py",
		    #Adding L1 step
    #"L1":"Configuration/PyReleaseValidation/Simulation.py",
		    #Adding DIGI2RAW step
    #"DIGI2RAW":"Configuration/PyReleaseValidation/Simulation.py",
		    #Adding HLT step
    #"HLT":"Configuration/PyReleaseValidation/Simulation.py",
		    }
#This option will not be used for now since the event content names keep changing, it can be edited by hand pre-release by pre-release if wanted (Still moved all to FEVTDEBUGHLT for now, except RECO, left alone).
EventContent={
	       #Use FEVTSIMDIGI for all steps but HLT and RECO
    "GEN,SIM":"FEVTDEBUGHLT",
    "DIGI":"FEVTDEBUGHLT",
	       #The following part should be edited to implement the wanted step and define the appropriate
	       #event content
	       #"L1":"FEVTDEBUGHLT",
	       #"DIGI2RAW":"FEVTDEBUGHLT",
	       #Use FEVTSIMDIGIHLTDEBUG for now
	       #"HLT":"FEVTDEBUGHLT",
	       #Use RECOSIM for RECO step
	       #"RAW2DIGI,RECO":"RECOSIM"
	       }
#The allowed profiles are:
AllowedProfile=[
	  "TimingReport",
	  "TimeReport",
	  "SimpleMemReport",
	  "EdmSize",
	  "IgProfperf",
	  "IgProfMemTotal",
	  "IgProfMemLive",
	  "IgProfMemAnalyse",
	  "valgrind",
	  "memcheck_valgrind",
	  "None"
	   ]
Profile = []
#Based on the profile code create the array of profiles to run:
for i in range(10):
    if str(i) in ProfileCode:
        firstCase = ((i==0)and((str(1) in ProfileCode)or(str(2) in ProfileCode)))  or  ((i==1)and(str(2) in ProfileCode))
        secCase   = ((i==5)and((str(6) in ProfileCode)or(str(7) in ProfileCode)))  or  ((i==6)and(str(7) in ProfileCode))
        if (firstCase or secCase):
	    Profile.append(AllowedProfile[i] +" @@@ reuse")
	else:
	    Profile.append(AllowedProfile[i])
#Hash for the profiler to run
Profiler={
    "TimingReport":"Timing_Parser",
    "TimingReport @@@ reuse":"Timing_Parser",#Ugly fix to be able to handle the reuse case
    "TimeReport":"Timereport_Parser",
    "TimeReport @@@ reuse":"Timereport_Parser",#Ugly fix to be able to handle the reuse case
    "SimpleMemReport":"SimpleMem_Parser",
    "EdmSize":"Edm_Size",
    "IgProfperf":"IgProf_perf.PERF_TICKS",
    "IgProfMemTotal":"IgProf_mem.MEM_TOTAL",
    "IgProfMemTotal @@@ reuse":"IgProf_mem.MEM_TOTAL",#Ugly fix to be able to handle the reuse case
    "IgProfMemLive":"IgProf_mem.MEM_LIVE",
    "IgProfMemLive @@@ reuse":"IgProf_mem.MEM_LIVE",#Ugly fix to be able to handle the reuse case
    "IgProfMemAnalyse":"IgProf_mem.ANALYSE",
    "valgrind":"ValgrindFCE",
    "memcheck_valgrind":"Memcheck_Valgrind",
    "None":"None"
     }
#Hash to switch from keyword to .cfi use of cmsDriver.py:
KeywordToCfi={
	       #For now use existing:
    "HZZLLLL":"H200ZZ4L.cfi",
	       #But for consistency we should add H190ZZ4LL.cfi into the Configuration/Generator/data:
	       #"HZZLLLL":"H190ZZ4L.cfi",
    "MINBIAS":"MinBias.cfi",
	       #For now test using existing (wrong but not to change rest of the code):
	       #"E -e 1000":"SingleElectronPt1000.cfi",
	       #But we'd like:
    "E -e 1000":"SingleElectronE1000.cfi",
    "MU- -e pt10":"SingleMuPt10.cfi",
	       #For now use existing (wrong but not to change rest of the code):
	       #"PI- -e 1000":"SinglePiPt1000.cfi",
	       #But we'd like:
    "PI- -e 1000":"SinglePiE1000.cfi",
    "TTBAR":"TTbar.cfi",
    "QCD -e 80_120":"QCD_Pt_80_120.cfi"
	    }
OutputStep = ""
for acandle in Candle:
    print "*Candle " + acandle
    stepIndex=0
    for step in steps:
	print >> simcandles, "#" + FileName[acandle]
	print >> simcandles, "#Step " + step
	print step
        if ("DIGI2RAW" in step):
	    #print "DIGI2RAW"
	    #print "$step;
	    SavedProfile=Profile
	    Profile=("None")
	if ("HLT" in step):
	    Profile=SavedProfile
	for prof in Profile:
            if ("EdmSize" in prof):
                #if ("GEN,SIM" in step): #Hack since we use SIM and not GEN,SIM extension (to facilitate DIGI)
                Command=FileName[acandle] + "_" + step + ".root "
		#    step="GEN\,SIM"
                #print "GEN\,SIM here"
	    else:
                if (not CustomiseFragment.has_key(step)):
		    #Temporary hack to have potentially added steps use the default Simulation.py fragment
		    #This should change once each group customises its customise python fragments.
		    CustomisePythonFragment=CustomiseFragment["DIGI"]
		else:
		    CustomisePythonFragment=CustomiseFragment[step]
		#Adding a fileout option too to avoid dependence on future convention changes in cmsDriver.py:
		OutputFileOption="--fileout=" + FileName[acandle] + "_" + step + ".root"
		OutputStep=step

		#Use --filein (have to for L1, DIGI2RAW, HLT) to add robustness
                if "GEN,SIM" in step: #there is no input file for GEN,SIM!
		    InputFileOption=""
		#Special hand skipping of HLT since it is not stable enough, so it will not prevent 
		#RAW2DIGI,RECO from running
                elif ("HLT" in steps[stepIndex-1]):
		    InputFileOption="--filein file:" + FileName[acandle] + "_" + steps[stepIndex-2] + ".root"
		else:
		    InputFileOption="--filein file:" + FileName[acandle] + "_" + steps[stepIndex-1] + ".root "
                if debug:
                    print InputFileOption, step, "GEN,SIM" in step, "HTL" in steps[stepIndex-1], steps
		#Adding .cfi to use new method of using cmsDriver.py
		#$Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile --customise=$CustomiseFragment{$step} ";
		#$Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=$step $InputFileOption $OutputFileOption --eventcontent=$EventContent{$step} --customise=$CustomiseFragment{$step} $cmsDriverOptions";
		Command=cmsDriver + " " + KeywordToCfi[acandle] + " -n " + NumberOfEvents + " --step=" + step + " " + InputFileOption + " " + OutputFileOption + " --customise=" + CustomisePythonFragment + " " + cmsDriverOptions
	    print >> simcandles, Command + " @@@ " + Profiler[prof] + " @@@ " + FileName[acandle] + "_" + OutputStep + "_" + prof
	stepIndex += 1
    #Add the extra "step" DIGI with PILE UP only for QCD_80_120:
#After digi pileup steps:
#Freeze this for now since we will only run by default the GEN-SIM,DIGI and DIGI pileup steps
AfterPileUpSteps=[
		   #"L1",
		   #"DIGI2RAW",
		   #"HLT",
		   #"RAW2DIGI,RECO"
		   ];
qcdStr = "QCD -e 80_120"
if ( qcdStr in Candle):
    thecandle = getFstOccur(qcdStr,Candle) 
	#First run the DIGI with PILEUP (using the MixingModule.py)
	#Hardcode stuff for this special step
    print >> simcandles, "#" + FileName[qcdStr]
    print >> simcandles, "#DIGI PILE-UP STEP"
    print "DIGI PILEUP";
    for prof in Profile:
        if ("EdmSize" in prof):
            Command=FileName[thecandle] + "_DIGI_PILEUP.root "
        else:
            InputFileOption="--filein file:" + FileName[thecandle]+"_GEN,SIM.root "
            OutputFileOption="--fileout="+FileName[thecandle]+"_DIGI_PILEUP.root"
		#Adding .cfi to use new method of using cmsDriver.py
		#$Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile --customise=$CustomiseFragment{$step} ";
		#$Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=DIGI $InputFileOption $OutputFileOption --PU --eventcontent=FEVTSIMDIGI --customise=Configuration/PyReleaseValidation/MixingModule.py $cmsDriverOptions";
            Command= cmsDriver + " " + KeywordToCfi[thecandle] + " -n "+ NumberOfEvents + " --step=DIGI " + InputFileOption + " " + OutputFileOption + " --PU --customise=Configuration/PyReleaseValidation/MixingModule.py " + cmsDriverOptions
        print >> simcandles, Command + " @@@ " + Profiler[prof] + " @@@ " + FileName[thecandle] + "_DIGI_PILEUP_" + prof
	#Very messy solution for now:
	#Setting the stepIndex variable to 2, i.e. RECO step
        stepIndex=2
        FileIn = {}
        FileIn["RECO"]="--filein file:"
        for step in AfterPileUpSteps:
	    print >> simcandles, "#" + FileName[thecandle]
	    print >> simcandles, "#Step " + step + " PILEUP";
	    print step + " PILEUP";
            if ("DIGI2RAW" in step):
		SavedProfile=Profile
		Profile=("None")
	    if ("HLT" in step):
		Profile=SavedProfile
	    
	    for prof in Profile:
                if ("EdmSize" in prof):
		    Command=FileName[thecandle] + "_" + step + "_PILEUP.root "
		else:
                    if (CustomiseFragment.has_key(step)):
			#Temporary hack to have potentially added steps use the default Simulation.py fragment
			#This should change once each group customises its customise python fragments.
			CustomisePythonFragment=CustomiseFragment["DIGI"]
		    else:
			CustomisePythonFragment=CustomiseFragment[step]
		    OutputStep=step + "_PILEUP";
		    InputFileOption=FileName[thecandle]+"_"+Steps[stepIndex-1]+"_PILEUP.root "
		    OutputFileOption="--fileout=" + FileName[candle] + "_"+step + "_PILEUP.root"
		    #Adding .cfi to use new method of using cmsDriver.py
		    #$Command="$cmsDriver $candle -n $NumberOfEvents --step=$step $FileIn{$step}$InputFile --customise=$CustomiseFragment{$step} ";
		    #$Command="$cmsDriver $KeywordToCfi{$candle} -n $NumberOfEvents --step=$step $InputFileOption $OutputFileOption --eventcontent=$EventContent{$step} --customise=$CustomiseFragment{$step} $cmsDriverOptions";
		    Command=cmsDriver + " " + KeywordToCfi[Candle] + " -n " + NumberOfEvents + " --step=" + step + " " + InputFileOption + " " + OutputFileOption + " --customise=" + CustomisePythonFragment + " " + cmsDriverOptions 
		print >> simcandles, Command + " @@@ " + Profiler[prof] + " @@@ " + FileName[Candle] + "_" + OutputStep + "_" + prof
	    stepIndex+=1

simcandles.close()
