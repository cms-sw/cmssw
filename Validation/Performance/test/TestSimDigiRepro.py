#!/usr/bin/env python
"""Script to test reproducibility at DIGI step

It runs the SIM step first, then it runs twice the DIGI step, dumps the DIGI contents
and compares them.

Usage: ./TestReproDIGI.py [options]

Options:
  -c ..., --candle=... specify a wanted candle (via .cfi)
  -n ...               specify the wanted number of events to run the test on
  -s ..., --skip=...   specify the number of events to skip when running the 2nd time
  --step=...           specify the step to test for reproducibility
  --cmsdriver=...      specify the necessary (release dependent) cmsDriver.py options
  -h, --help           show this help
  -d                   show debugging information while parsing
  --no_exec            print out the commands the script would have launched.

Examples:
  TestSimDigiRepro.py                  Defaults to TTbar.cfi
  TestSimDigiRepro.py -c MinBias.cfi
  TestSimDigiRepro.py --step=DIGI

Note:
  This script relies on code in the following packages:
    -SimG4Core/Application [/test]
    -Validation/Performance [/test]
    -Configuration/PyReleaseValidation [/python]
  
  """
#Future improvements for style, speed and robustness:
#1-use links to original files in the release or base instead of copying
#2-use try/except to check for file formats
#3-use regular expressions to be independent of whitespace formatting and potential future changes in the template
#configuration rootfile name

import os
#Let's add the environment variables we'll need
cmssw_base=os.environ["CMSSW_BASE"]
print cmssw_base
cmssw_release_base=os.environ["CMSSW_RELEASE_BASE"]
print cmssw_release_base
#Lets enrich our script with options!
import sys
#Let's add the local directory to the PYTHONPATH, so that we can play with customise fragments locally:
#sys.path.append(os.getcwd())
#print "Current PYTHONPATH is %s"%os.environ["PYTHONPATH"]
import getopt

def usage():
    print __doc__

def GetFile(myFile,myFileLocation):
    print "Copying over %s" % myFile
    sys.stdout.flush()
    #First try the CMSSW_BASE version if there
    if os.access(cmssw_base+myFileLocation+myFile,os.F_OK):
        if "runSimDigiDumper" in myFile:
            #in this case dump it in the current directory
            CpFile="cp -pR "+cmssw_base+myFileLocation+myFile+" ."
        else:
            #For python fragments instead dump them in $CMSSW_BASE/python
            CpFile="cp -pR "+cmssw_base+myFileLocation+myFile+" "+cmssw_base+"/python/."
        if noexec:
            print "**%s"%CpFile
        else:
            os.system(CpFile)
    #If not there try the CMSSW_RELEASE_BASE version
    else:
        print "File %s not found in %s" % (myFile,cmssw_base+myFileLocation)
        print "Trying in %s" % cmssw_release_base+myFileLocation
        if os.access(cmssw_release_base+myFileLocation+myFile,os.F_OK):
            CpFile="cp -pR "+cmssw_release_base+myFileLocation+myFile+" ."
            if noexec:
                print "**%s"%CpFile
            else:
                os.system(CpFile)
        else:
            print "**COULD NOT FIND THE NECESSARY %s FILE!**" % myFile
    sys.stdout.flush()
    
def RestoreSkipEventSetting(myRestoreFragment,mySkipEvents):
    NewSkipEvents="skipEvents=cms.untracked.uint32("+str(mySkipEvents)+")"
    NewFragmentFilename=cmssw_base+"/python/"+myRestoreFragment.split(".")[0]+"Skip"+str(mySkipEvents)+"Evts.py"
    NewFragmentFile=open(NewFragmentFilename,"w")
    NewFragmentFileContent=""
    LocalRestoreFragment=cmssw_base+"/python/"+myRestoreFragment
    if mySkipEvents != 3: #Avoid doing this if there is no need, default is 3 in the python fragment
        for line in open(LocalRestoreFragment,"r").readlines():
            #This part of the code relies on the string to not be changed in the template cfg!
            #A weak point...
            newline=line.replace("skipEvents=cms.untracked.uint32(3)",NewSkipEvents)
            NewFragmentFileContent+=newline
    else:
        NewFragmentFileContent=open(LocalRestoreFragment,"r").read()
      
    NewFragmentFile.write(NewFragmentFileContent)
    return NewFragmentFilename.split("/")[-1] #Return only the name of the file, no path

def MakeCfg(myCfgFile,myOldRootFile,myNewRootFile):
    
    NewCfgFileName=myCfgFile.split(".")[0]+myNewRootFile.split("_")[0]+myNewRootFile.split("_")[1]+myNewRootFile.split("_")[2].split(".")[0]+"."+myCfgFile.split(".")[1]
    NewCfgFile=open(NewCfgFileName,"w")
    NewCfgFileContent=""
    for line in open(myCfgFile,"r").readlines():
        newline=line.replace(myOldRootFile,myNewRootFile)
        NewCfgFileContent+=newline
    NewCfgFile.write(NewCfgFileContent)
    return NewCfgFileName

def CompareDumps(myDumpType,myCandle,myStep,mySkipEvents):
    print "Comparing %s for %s after skipping %s" % (myDumpType,myCandle,mySkipEvents)
    mySavedSeedsFile=myDumpType+myCandle.split(".")[0]+myStep+"SavedSeeds.log"
    myRestoredSeedsFile=myDumpType+myCandle.split(".")[0]+myStep+"RestoredSeeds.log"
    Event=0
    FirstGoodEvent=0
    DifferentLines=0
    lineSavedSeeds=iter(open(mySavedSeedsFile,"r"))
    lineRestoredSeeds=iter(open(myRestoredSeedsFile,"r"))
    FirstGoodEvent=str(int(mySkipEvents)+1)
    EventNumber=" Event %s" % FirstGoodEvent
    #Positioning the two files iterators to the right position before comparison
    for line in lineSavedSeeds:
        if line.rfind(EventNumber)>0:
            Event=line.split()[8][:-1]#ARGH... they added a , in the format!
            if Event==FirstGoodEvent:
                print "Synchronized file %s to beginning of event %s" % (mySavedSeedsFile,Event)
                break
    for line in lineRestoredSeeds:
        if line.rfind(EventNumber)>0:
            Event=line.split()[8][:-1]#ARGH... they added a , in the format!
            if Event==FirstGoodEvent:
                print "Synchronized file %s to beginning of event %s" % (myRestoredSeedsFile,Event)
                break
    #Now both iterators indeces should be synchronized, so let's compare line by line:
    for l1,l2 in zip(lineSavedSeeds,lineRestoredSeeds):
        if l1.startswith("Begin processing the "):
            Event=l1.split()[8][:-1]#ARGH... they added a , in the format!
            continue
        if l1.startswith("%MSG") or l1.startswith("Info") or l1.rfind("FwkReport")>0 or l1.rfind("Root_Information")>0:
            continue
        if l1 != l2:
            DifferentLines+=1
            print "The following two lines are different in the two files above (event %s):"% Event
            print "< %s" % l1
            print "> %s" % l2
        if DifferentLines>=50:
            print "***There are more than 50 different lines in the files %s and %s, you should inspect them!***" % (mySavedSeedsFile,myRestoredSeedsFile)
            break
    if DifferentLines==0:
        print "The content dumped in %s and %s is identical" % (mySavedSeedsFile,myRestoredSeedsFile)
    sys.stdout.flush()
def ExecuteStartingCommand(myStep,candle,numEvents,cmsdriveroptions):
    #Complicated handling of special case introduced by GEN:ProductionFilterSequence
    if ":" in StartingSteps[myStep]:
        if "," in StartingSteps[myStep]:
            FirstStep=StartingSteps[myStep].split(":")[0]
            SecondStep=StartingSteps[myStep].split(",")[1]
            SpecialStep=FirstStep+","+SecondStep
        else:
            SpecialStep=StartingSteps[myStep].split(":")[0]
        
        myFile=candle.split('.')[0]+"_"+SpecialStep
    else:
        myFile=candle.split('.')[0]+"_"+StartingSteps[myStep]
    myCommand="cmsDriver.py "+candle+" -n "+str(numEvents)+" -s "+StartingSteps[myStep]+" --customise="+CustomiseFiles[StartingSteps[myStep]]+" "+cmsdriveroptions+" --fileout="+myFile+".root>& "+myFile+".log"
    print "Executing starting command:\n%s" % myCommand
    sys.stdout.flush()
    if noexec:
        print "**%s"%myCommand
    else:
        ExitCode=os.system(myCommand)
        if ExitCode != 0:
            print "Exit code for %s was %s" % (myCommand, ExitCode)
            ExitCode=0
        sys.stdout.flush()
    return(myFile)

def TestRepro(myStep,myInputFile,candle,numEvents,skipEvents,cmsdriveroptions):
    #First round saving seeds:
    mySavedSeedsFile=candle.split('.')[0]+"_"+myStep+"_SavedSeeds"
    mySaveSeedsCommand="cmsDriver.py "+candle+" -n "+str(numEvents)+" -s "+myStep+" --customise="+CustomiseFiles[myStep][0]+" "+cmsdriveroptions+" --filein file:"+myInputFile+".root --fileout="+mySavedSeedsFile+".root >& "+mySavedSeedsFile+".log"
    print "Executing %s step, saving random seeds with command:\n%s" % (myStep,mySaveSeedsCommand)
    sys.stdout.flush()
    if noexec:
        print "**%s"%mySaveSeedsCommand
    else:
        ExitCode=os.system(mySaveSeedsCommand)
        if ExitCode != 0:
            print "Exit code for %s was %s" % (mySaveSeedsCommand, ExitCode)
            ExitCode=0
        sys.stdout.flush()
    #Second round restoring seeds:
    #For SaveRandomSeeds.py it's OK to use the version in the release
    #but for RestoreRandomSeeds.py we want to access the number of events to skip
    #Get the RestoreRandomSeeds.py locally (it could be done without copying the file locally, just using the path...)
    RestorePy=CustomiseFiles[myStep][1].split("/")[2]
    GetFile(RestorePy,"/src/Validation/Performance/python/")
    
    #Edit the fragment to start from the wanted event:
    NewRestorePy=RestoreSkipEventSetting(RestorePy,skipEvents)

    #Run the step restoring seeds
    myRestoredSeedsFile=candle.split('.')[0]+"_"+myStep+"_RestoredSeeds_Skip"+str(skipEvents)+"Evts"
    myRestoreSeedsCommand="cmsDriver.py "+candle+" -n "+str(numEvents)+" -s "+myStep+" --customise="+NewRestorePy+" "+cmsdriveroptions+" --filein file:"+mySavedSeedsFile+".root --fileout="+myRestoredSeedsFile+".root >& "+myRestoredSeedsFile+".log"
    print "Executing %s step, restoring random seeds with command:\n%s" % (myStep,myRestoreSeedsCommand)
    sys.stdout.flush()
    if noexec:
        print "**%s"%myRestoreSeedsCommand
    else:
        ExitCode=os.system(myRestoreSeedsCommand)
        if ExitCode != 0:
            print "Exit code for %s was %s" % (myRestoreSeedsCommand, ExitCode)
            ExitCode=0
        sys.stdout.flush()
    return(mySavedSeedsFile,myRestoredSeedsFile)

def main(argv):
    #Let's define some defaults: 
    candle = "TTbar.cfi"
    numEvents = 10
    skipEvents = 3
    step="SIM"
    #cmsdriveroptions for CMSSW_3_1_0_pre2/pre3
    cmsdriveroptions="--eventcontent FEVTDEBUG --conditions FrontierConditions_GlobalTag,IDEAL_30X::All"
    #noexec=False
    global noexec
    #Let's check the command line arguments
    try:                                
        opts, args = getopt.getopt(argv, "c:n:s:hd", ["candle=","cmsdriver=","skip=","step=","no_exec","help"])
    except getopt.GetoptError:
        print "This argument option is not accepted"
        usage()                          
        sys.exit(2)       
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt == '-d':
            global _debug
            _debug = 1
        elif opt in ("-c", "--candle"):
            candle = arg
            print "You chose to run on %s" % candle
        elif opt == "-n":
            numEvents = arg 
            print "You chose to run the test on %s events" % numEvents
        elif opt in ("-s", "--skip"):
            skipEvents = arg
            if int(skipEvents) >= int(numEvents):
                print "You chose a number of events to skip larger or equal than the number of events!"
                sys.exit()
            print "You chose to run the test skipping %s events the second time" % skipEvents
        elif opt == "--step":
            step = arg
            print "You chose to run the test on %s step" % step
        elif opt == "--cmsdriver":
            cmsdriveroptions = arg
            print "You chose to run the test with cmsdriver options %s " % cmsdriveroptions
        elif opt == "--no_exec":
            noexec=True
            print "You chose to print out the script commands without executing them"
    #Case with no arguments (using defaults)
    if opts == []:
        print "No arguments given, so DEFAULT test will be run:"
    else:
        print "Following options will be used for this test:"
    #Print options out anyway!
    print "Step: %s " % step
    print "Candle: %s " % candle
    print "Number of Events: %s " % numEvents
    print "Number of Events to skip on the second run: %s" % skipEvents
    sys.stdout.flush()
    #Let's do the tests!
    #Depending on the step launch the starting command to provide the input file for the step to be tested
    InputFile=ExecuteStartingCommand(step,candle,numEvents,cmsdriveroptions)
    #Depending on the step launch the actual reproducibility test, saving and restoring seeds:
    (SavedSeedsFile,RestoredSeedsFile)=TestRepro(step,InputFile,candle,numEvents,skipEvents,cmsdriveroptions)
    
    #Get the cfg files necessary to run the dumpers(use python version of them!):
    CfgFilesLocation={
        "SIM":{"runSimHitCaloHitDumper_cfg.py":"/src/SimG4Core/Application/test/",
               "runSimTrackSimVertexDumper_cfg.py":"/src/SimG4Core/Application/test/"},
        "DIGI":{"runSimDigiDumper_cfg.py":"/src/Validation/Performance/test/"}
        }
    
    for CfgFile in CfgFilesLocation[step].keys():
        GetFile(CfgFile,CfgFilesLocation[step][CfgFile])

    #Modify the cfgs as needed 1 copy of each pointing to the SIMSavedSeedsFile, 1 copy of each pointing to the SIMRestoredSeedsFile:
    #Function MakeCfg takes the cfg, the default root filename and the new root filename to point to,
    #replaces the root filename, saves a new cfg file with and obvious filename.
    #At the moment we are assuming the name of these files to be always myfile.root, should make this more robust
    #and read it directly from the file!
    
    RootFiles=[SavedSeedsFile+".root",RestoredSeedsFile+".root"]
    
    for CfgFile in CfgFilesLocation[step].keys():
        for RootFile in RootFiles: 
            Cfg=MakeCfg(CfgFile,"myfile.root",RootFile)
            DumpFile=Cfg.split(".")[0]+".log"
            DumperCommand="cmsRun "+Cfg+" >& "+DumpFile
            print "Executing %s" % DumperCommand
            if noexec:
                print "**Did not execute the command (--no_exec option selected)"
            else:
                ExitCode=os.system(DumperCommand)
                if ExitCode != 0:
                    print "Exit code for %s was %s" % (DumperCommand, ExitCode)
                    ExitCode=0
                sys.stdout.flush()
            
    #Do the comparisons:
    #Use the function CompareDumps that skips to the wanted event before starting comparison
    #and excludes a number or irrelevant lines from the comparison.

    Tests={
        "SIM":["runSimHitCaloHitDumper","runSimTrackSimVertexDumper"],
        "DIGI":["runSimDigiDumper"]
        }

    for Test in Tests[step]:
        DumpType=Test+"_cfg"
        CompareDumps(DumpType,candle,step,skipEvents)
        
    
if __name__ == "__main__":
    #Some needed global variables
    noexec=False
    StartingSteps={"SIM":"GEN:ProductionFilterSequence","DIGI":"GEN:ProductionFilterSequence,SIM"}
    CustomiseFiles={
        "GEN:ProductionFilterSequence":"Validation/Performance/TimeMemoryInfo.py",
        "GEN,SIM":"Validation/Performance/TimeMemoryG4Info.py",
        "GEN:ProductionFilterSequence,SIM":"Validation/Performance/TimeMemoryG4Info.py",
        "SIM":["Validation/Performance/SaveRandomSeedsSim.py","Validation/Performance/RestoreRandomSeedsSim.py"],
        "DIGI":["Validation/Performance/SaveRandomSeedsDigi.py","Validation/Performance/RestoreRandomSeedsDigi.py"]
        }
    main(sys.argv[1:])
#os.system("cmsDriver.py pippo")
#usage()
