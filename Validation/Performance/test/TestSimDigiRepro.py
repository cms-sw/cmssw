#!/usr/bin/env python
"""Script to test reproducibility at DIGI step

It runs the SIM step first, then it runs twice the DIGI step, dumps the DIGI contents
and compares them.

Usage: ./TestReproDIGI.py [options]

Options:
  -c ..., --candle=... specify a wanted candle (via .cfi)
  -n ...               specify the wanted number of events to run the test on
  -s ..., --skip=...   specify the number of events to skip when running the 2nd time 
  -h, --help           show this help
  -d                   show debugging information while parsing

Examples:
  TestReproDIGI.py                  Defaults to TTbar.cfi
  TestReproDIGI.py -c MinBias.cfi

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
cmssw_release_base=os.environ["CMSSW_RELEASE_BASE"]
#Lets enrich our script with options!
import sys
import getopt

def usage():
    print __doc__

def GetFile(myFile,myFileLocation):
    print "Copying over %s" % myFile
    #First try the CMSSW_BASE version if there
    if os.access(cmssw_base+myFileLocation+myFile,os.F_OK):
        CpFile="cp -pR "+cmssw_base+myFileLocation+myFile+" ."
        os.system(CpFile)
    #If not there try the CMSSW_RELEASE_BASE version
    else:
        print "File %s not found in %s" % (myFile,cmssw_base+myFileLocation)
        print "Trying in %s" % cmssw_release_base+myFileLocation
        if os.access(cmssw_release_base+myFileLocation+myFile,os.F_OK):
            CpFile="cp -pR "+cmssw_release_base+myFileLocation+myFile+" ."
            os.system(CpFile)
        else:
            print "**COULD NOT FIND THE NECESSARY %s FILE!**" % myFile
    
def RestoreSkipEventSetting(myRestoreFragment,mySkipEvents):
    NewSkipEvents="skipEvents=cms.untracked.uint32("+str(mySkipEvents)+")"
    NewFragmentFilename=myRestoreFragment.split(".")[0]+"Skip"+str(mySkipEvents)+"Evts.py"
    NewFragmentFile=open(NewFragmentFilename,"w")
    NewFragmentFileContent=""
    if mySkipEvents != 3: #Avoid doing this if there is no need, default is 3 in the python fragment
        for line in open(myRestoreFragment,"r").readlines():
            #This part of the code relies on the string to not be changed in the template cfg!
            #A weak point...
            newline=line.replace("skipEvents=cms.untracked.uint32(3)",NewSkipEvents)
            NewFragmentFileContent+=newline
    else:
        NewFragmentFileContent=open(myRestoreFragment,"r").read()
      
    NewFragmentFile.write(NewFragmentFileContent)
    return NewFragmentFilename

def MakeCfg(myCfgFile,myOldRootFile,myNewRootFile):
    NewCfgFileName=myCfgFile.split(".")[0]+myNewRootFile.split("_")[0]+myNewRootFile.split("_")[3].split(".")[0]+"."+myCfgFile.split(".")[1]
    NewCfgFile=open(NewCfgFileName,"w")
    NewCfgFileContent=""
    for line in open(myCfgFile,"r").readlines():
        newline=line.replace(myOldRootFile,myNewRootFile)
        NewCfgFileContent+=newline
    NewCfgFile.write(NewCfgFileContent)
    return NewCfgFileName

def CompareDumps(myDumpType,myCandle,mySkipEvents):
    print "Comparing %s for %s after skipping %s" % (myDumpType,myCandle,mySkipEvents)
    mySavedSeedsFile=myDumpType+myCandle.split(".")[0]+"SavedSeeds.log"
    myRestoredSeedsFile=myDumpType+myCandle.split(".")[0]+"RestoredSeeds.log"
    Event=0
    FirstGoodEvent=0
    Different=True
    lineSavedSeeds=iter(open(mySavedSeedsFile,"r"))
    lineRestoredSeeds=iter(open(myRestoredSeedsFile,"r"))
    FirstGoodEvent=str(int(mySkipEvents)+1)
    EventNumber=" Event %s" % FirstGoodEvent
    #Positioning the two files iterators to the right position before comparison
    for line in lineSavedSeeds:
        if line.rfind(EventNumber)>0:
            Event=line.split()[8]
            if Event==FirstGoodEvent:
                print "Synchronized file %s to beginning of event %s" % (mySavedSeedsFile,Event)
                break
    for line in lineRestoredSeeds:
        if line.rfind(EventNumber)>0:
            Event=line.split()[8]
            if Event==FirstGoodEvent:
                print "Synchronized file %s to beginning of event %s" % (myRestoredSeedsFile,Event)
                break
    #Now both iterators indeces should be synchronized, so let's compare line by line:
    for l1,l2 in zip(lineSavedSeeds,lineRestoredSeeds):
        if l1.startswith("Begin processing the ") or l1.startswith("%MSG") or l1.startswith("Info") or l1.rfind("FwkReport")>0 or l1.rfind("Root_Information")>0:
            continue
        if l1 != l2:
            Different=True
            print "The following two lines are different in the two files above:"
            print "< %s" % l1
            print "> %s" % l2
        else:
            Different=False
    if not Different:
        print "The content dumped in %s and %s is identical" % (mySavedSeedsFile,myRestoredSeedsFile)
    
def main(argv):
    #Let's define some defaults:
    candle = "TTbar.cfi"
    numEvents = 10
    skipEvents = 3

    #Let's check the command line arguments
    try:                                
        opts, args = getopt.getopt(argv, "c:n:s:hd", ["candle=","skip=","help"])
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
    #Case with no arguments (using defaults)
    if opts == []:
        print "No arguments given, so DEFAULT test will be run:"
        print "Candle: %s " % candle
        print "Number of Events: %s " % numEvents
        print "Number of Events to skip on the second run: %s" % skipEvents

    #Let's do the tests!
    
    #First we need to generate the events:
    GENFile=candle.split('.')[0]+"_GEN"
    GENCommand="cmsDriver.py "+candle+" -n "+numEvents+" -s GEN --customise=Configuration/PyReleaseValidation/Simulation.py --fileout="+GENFile+".root>& "+GENFile+".log"
    print "Executing %s" % GENCommand
    ExitCode=os.system(GENCommand)
    if ExitCode != 0:
        print "Exit code for %s was %s" % (GENCommand, ExitCode)
        ExitCode=0
        
    #For SaveRandomSeeds.py it's OK to use the version in the release
    #but for RestoreRandomSeeds.py we want to access the number of events to skip
    
    #First round SIM+DIGI saving seeds:
    SIMDIGISavedSeedsFile=candle.split('.')[0]+"_SIM_DIGI_SavedSeeds"
    SIMDIGISaveSeedsCommand="cmsDriver.py "+candle+" -n "+numEvents+" -s SIM --customise=Configuration/PyReleaseValidation/SaveRandomSeeds.py --filein file:"+GENFile+".root --fileout="+SIMDIGISavedSeedsFile+".root >& "+SIMDIGISavedSeedsFile+".log"
    print "Executing %s" % SIMDIGISaveSeedsCommand
    ExitCode=os.system(SIMDIGISaveSeedsCommand)
    if ExitCode != 0:
        print "Exit code for %s was %s" % (SIMDIGISaveSeedsCommand, ExitCode)
        ExitCode=0
        
    #Second round SIM+DIGI restoring seeds:
    #Get the RestoreRandomSeeds.py locally (it could be done without copying the file locally, just use the link...)
    RestorePy="RestoreRandomSeeds.py"
    GetFile(RestorePy,"/src/Configuration/PyReleaseValidation/python/")

    #Edit the fragment to start from the wanted event:
    NewRestorePy=RestoreSkipEventSetting(RestorePy,skipEvents)

    #Run the SIM+DIGI restoring seeds
    SIMDIGIRestoredSeedsFile=candle.split('.')[0]+"_SIM_DIGI_RestoredSeeds_Skip"+str(skipEvents)+"Evts"
    SIMDIGIRestoreSeedsCommand="cmsDriver.py "+candle+" -n "+numEvents+" -s SIM --customise="+NewRestorePy+" --filein file:"+SIMDIGISavedSeedsFile+".root --fileout="+SIMDIGIRestoredSeedsFile+".root >& "+SIMDIGIRestoredSeedsFile+".log"
    print "Executing %s" % SIMDIGIRestoreSeedsCommand
    ExitCode=os.system(SIMDIGIRestoreSeedsCommand)
    if ExitCode != 0:
        print "Exit code for %s was %s" % (SIMDIGIRestoreSeedsCommand, ExitCode)
        ExitCode=0
    
    #Get the cfg files necessary to run the dumpers(use python version of them!):
    CfgFilesLocation={"runSimHitCaloHitDumper_cfg.py":"/src/SimG4Core/Application/test/","runSimTrackSimVertexDumper_cfg.py":"/src/SimG4Core/Application/test/","runSimDigiDumper_cfg.py":"/src/Validation/Performance/test/"}
    
    for CfgFile in CfgFilesLocation.keys():
        GetFile(CfgFile,CfgFilesLocation[CfgFile])

    #Modify the cfgs as needed 1 copy of each pointing to the SIMDIGISavedSeedsFile, 1 copy of each pointing to the SIMIDIGIRestoredSeedsFile:
    #Function MakeCfg takes the cfg, the default root filename and the new root filename to point to,
    #replaces the root filename, saves a new cfg file with and obvious filename.
    #At the moment we are assuming the name of these files to be always myfile.root, should make this more robust
    #and read it directly from the file!
    
    RootFiles=[SIMDIGISavedSeedsFile+".root",SIMDIGIRestoredSeedsFile+".root"]
    
    for CfgFile in CfgFilesLocation.keys():
        for RootFile in RootFiles: 
            Cfg=MakeCfg(CfgFile,"myfile.root",RootFile)
            DumpFile=Cfg.split(".")[0]+".log"
            DumperCommand="cmsRun "+Cfg+" >& "+DumpFile
            print "Executing %s" % DumperCommand
            ExitCode=os.system(DumperCommand)
            if ExitCode != 0:
                print "Exit code for %s was %s" % (DumperCommand, ExitCode)
                ExitCode=0
            
            
    #Do the comparisons:
    #Use the function CompareDumps that skips to the wanted event before starting comparison
    #and excludes a number or irrelevant lines from the comparison.

    Tests=["runSimHitCaloHitDumper","runSimTrackSimVertexDumper","runSimDigiDumper"]

    for Test in Tests:
        DumpType=Test+"_cfg"
        CompareDumps(DumpType,candle,skipEvents)
        
    
if __name__ == "__main__":
    main(sys.argv[1:])
#os.system("cmsDriver.py pippo")
#usage()
