#!/usr/bin/env cmsRun

import shutil
import sys

from Validation.RecoTau.ValidationOptions_cff import *

process = cms.Process("TEST")

# command options defined in Validation/RecoTau/python/ValidationOptions_cfi
options.parseArguments()

checkOptionsForBadInput()

if not calledBycmsRun() and not options.gridJob:
   print "Run 'cmsRun RunTauValidation_cfg.py (elp' for options."
   # quit here so we dont' create a bunch of directories
   #  if the user only wants the help
   #sys.exit()


# DQM store, PDT sources etc
process.load("DQMServices.Components.EDMtoMEConverter_cff")

######################################
#                                    #
#       Output Info Store            #
#                                    #
######################################

"""
   Data is stored in

   TauID/[EventType]_[DataSource]_[Conditions][label]

"""

#outputDirName = "Validation_%s" % ReleaseVersion
outputDirName = "TauID"


outputDir = os.path.join(os.getcwd(), outputDirName) 
# This is the directory where we store the stuff about our current configuration
outputBaseDir = outputDir
subDirName = ""
subDirName += "%s_%s" % (options.eventType, "DQM") #Only "recoFiles" options sice we will get the information needen from official relVal DQM stream
outputDir = os.path.join(outputDir, subDirName)

# Store configuration, showtags, etc in a sub directory
configDir = os.path.join(outputDir, "Config")

if os.path.exists(outputDir) and options.batchNumber < 0:# and not options.gridJob:
   print "Output directory %s already exists!  OK to overwrite?" % outputDir
   while True:
      input = raw_input("Please enter [y/n] ")
      if (input == 'y'):
         break
      elif (input == 'n'):
         print " ...exiting."
         sys.exit()

if not os.path.exists(outputDir):
   os.makedirs(outputDir)

if not os.path.exists(configDir):
   os.makedirs(configDir)

######################################
#                                    #
#       Data Source Setup            #
#                                    #
######################################

process.schedule = cms.Schedule()

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

import Validation.RecoTau.DBSApi_cff as mydbs
print "Accessing DBS to retrieve the input files..."
mydbs.FillSource(options.eventType+'DQM',process.source)

if len(process.source.fileNames) == 0:
    print "No"
    sys.exit(0)

print process.source

outputFileName = os.path.join(configDir, "DataSource_cff.py")
outputFile = open(outputFileName,'w')
outputFile.write('import FWCore.ParameterSet.Config as cms\n')
outputFile.write('source = %s\n'%process.source)

# have to set max events here, since it may get written by the 
# dataSource cffs
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)


######################################
#                                    #
#       Validation Setup             #
#                                    #
######################################

# Store the tags and CVS diff to the tags, and the current release
#  only do this once in a batch job.  The additional tar file is a fail safe - 
#  the parameters shouldn't change in outputDir.
if (options.batchNumber <= 0 ):#and not options.gridJob):
   os.system("cd $CMSSW_BASE/src; \
              showtags -t -r > showtags.txt; \
              cvs -q diff >& diffToTags.patch;\
              cvs -q diff -r %s >& diffToVanillaRelease.patch; \
              tar -cvzf TagsAndDiff.tar.gz showtags.txt *.patch; \
              mv showtags.txt *.patch %s; \
              mv TagsAndDiff.tar.gz %s" % (ReleaseVersion, configDir, configDir))

#Validation output file
outputFileNameBase = "TauVal_%s" % ReleaseVersion
if options.label != "none":
   outputFileNameBase += "_%s" % options.label
outputFileNameBase += "_"
outputFileNameBase += options.eventType
outputFileNameBase += '.root'
outputFileName      = os.path.join(outputDir, outputFileNameBase)

print 'The output file will be: '+outputFileName

process.saveTauEff = cms.EDAnalyzer("TauDQMSimpleFileSaver",
  outputFileName = cms.string(outputFileName),
  rootDirectory  = cms.string("RecoTauV"),
  pattern        = cms.string("*RecoTauV/*%s_*" % options.eventType.replace('FastSim','')),
)

print process.saveTauEff.pattern
process.getHistos = cms.Path( process.EDMtoMEConverter*process.saveTauEff )#getattr(process,'produceDenominator'+options.eventType) )

process.schedule.append(process.getHistos)

######################################
#                                    #
#       CFG dump                     #
#                                    #
######################################

dumpFileName = "cfgDump"
dumpFileName += ".py"
processDumpFile = open('%s/%s' % (configDir, dumpFileName), 'w')
print >> processDumpFile, process.dumpPython()

######################################
#                                    #
#       Histo storage                #
#                                    #
######################################

from ROOT import *

TargetDir = "RecoTauV"
OutFileName = TargetDir+"_"+options.eventType+".root"
InputFileName = outputFileName

#opening output file
outFile = TFile(OutFileName,"recreate")
#opening input file
fullFile = TFile(InputFileName)

#retrieving interesting directory position
source = gDirectory.ls()
next=TIter(fullFile.GetListOfKeys())
dir0 = None
dir1 = None
for key in next:
     cl = gROOT.GetClass(key.GetClassName())
     if(cl.InheritsFrom("TDirectory")):
        dir0_ = key.ReadObj()
        dir0 = dir0_.GetName()
        next2=TIter(dir0_.GetListOfKeys())
        for key in next2:
           cl2 = gROOT.GetClass(key.GetClassName())
           if(cl2.InheritsFrom("TDirectory")):
              dir1_=key.ReadObj()
              if TargetDir in dir1_.GetName():
                 dir1 = dir1_.GetName()

#entering inside target directory
InputDir = dir0+"/"+dir1
gDirectory.cd(InputDir)

#Listing subdirectories of TargetDir
SubDirs = []
next3=TIter(gDirectory.GetListOfKeys())
for key in next3:
   cl3 = gROOT.GetClass(key.GetClassName())
   if(cl3.InheritsFrom("TDirectory")):
      dir2_ = key.ReadObj()
      dir2 = dir2_.GetName()
      if options.eventType in dir2:
         SubDirs.append(dir2)
#Writing objects to file
for sub_key in SubDirs:
   outFile.cd()
   gDirectory.mkdir(sub_key)
   fullFile.cd()
   gDirectory.cd(InputDir+"/"+sub_key)
   hList = TIter(gDirectory.GetListOfKeys())
   for h_key in hList:
      cl4 = gROOT.GetClass(h_key.GetClassName())
      if(cl4.InheritsFrom("TH1F")):
         outFile.cd()
         gDirectory.cd(sub_key)
         hist = h_key.ReadObj()
         hist.Write()

outFile.Close()









