#!/usr/bin/env python
"""

     MergeFilesAndCalculateEfficiencies.py

        Merges multiple root files containing the numerator and denominator histograms produced by the
        TauTagValidation package.  The efficiency (num/denominator) is then computed 
        as defined Validation/RecoTau/python/RecoTauValidation_cff.py and stored in OutputFile_Eff.root

     Usage: cmsRun MergeFilesAndCalculateEfficiencies.py OutputFile InputFiles

     Example: ./MergeFilesAndCalculateEfficiencies.py CMSSW_3_1_0_Signal.root CMSSW_3_1_0_ZTT_*.root

"""

import os
import sys
import FWCore.ParameterSet.Config as cms

if len(sys.argv) < 4:
   print "Error. Expected at least 2 arguments\n\nUsage: MergeFilesAndCalculateEfficiencies.py OutputFile InputFileGlob"
   sys.exit()

OutputFile = sys.argv[2]
Inputs     = sys.argv[3:]

for aFile in Inputs:
   if not os.path.exists(aFile):
      print "Input file %s does not exist!" % aFile
      sys.exit()

if os.path.exists(OutputFile):
   GotGoodValue = False
   userInput = ""
   while not GotGoodValue:
      userInput = raw_input("Output file %s exists; replace it? [yn] " % OutputFile).strip()
      if userInput != 'y' and userInput != 'n':
         print "Please enter y or n"
      else:
         GotGoodValue = True
   if userInput == 'n':
      sys.exit()

# Merge files using hadd utility
commandString  = "hadd -f %s " % OutputFile
for aFile in Inputs:
   commandString += aFile
   commandString += " "

os.system(commandString)

print "Running cmsRun command to generate efficiencies"

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.DQMStore = cms.Service("DQMStore")
process.load("Validation.RecoTau.RecoTauValidation_cfi")

process.loadFile   = cms.EDAnalyzer("DQMFileLoader",
      myFiles = cms.PSet(
         inputFileNames = cms.vstring(OutputFile),
         scaleFactor = cms.double(1.),
         )
)

process.saveTauEff = cms.EDAnalyzer("DQMSimpleFileSaver",
#  outputFileName = cms.string(OutputFile.replace('.root', '_Eff.root'))
  outputFileName = cms.string(OutputFile)
)

process.p = cms.Path(
      process.loadFile*
      process.TauEfficiencies*
      process.saveTauEff
      )

