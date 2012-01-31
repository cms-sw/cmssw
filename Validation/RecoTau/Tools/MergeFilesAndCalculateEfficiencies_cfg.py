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
import glob
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

#options = VarParsing.VarParsing ('standard')
options = VarParsing.VarParsing ()
options.register( 'out', 'Same name of the fisrt input file + _Eff', VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Sets the name of the output file")
options.register( 'type', "ZTT", VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Sets the type of events to which calculate efficiency, Available types [ZTT, ZMM, ZEE, QCD]")
options.register( 'inputs', [],VarParsing.VarParsing.multiplicity.list, VarParsing.VarParsing.varType.string, "Sets the input file names")

options.parseArguments()
Inputs = []
for entry in options.inputs:
   Inputs.extend(glob.glob(entry) )

if len(Inputs) == 0:
   print 'No inputs provided! Exiting...'
   sys.exit(0)

if not options.type in ['ZTT', 'ZMM', 'ZEE', 'QCD']:
   print 'Inexisting event type! Exiting...'
   sys.exit(0)

if options.out == 'Same name of the fisrt input file + _Eff':
   OutputFile = Inputs[0][0:Inputs[0].find('.root')]+'_Eff.root'
else:
   OutputFile = options.out


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
    input = cms.untracked.int32(0)
)

process.DQMStore = cms.Service("DQMStore")
process.load("Validation.RecoTau.ValidateTausOn%s_cff" % options.type)

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
      getattr(process,'TauEfficiencies%s' % options.type)*
      process.saveTauEff
      )

