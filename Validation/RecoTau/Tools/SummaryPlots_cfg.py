import FWCore.ParameterSet.Config as cms
import os
import glob

try:
   ReleaseBase = os.path.join(os.environ['CMSSW_BASE'], "src")
   ReleaseVersion = os.environ['CMSSW_VERSION']
except KeyError:
   print "CMSSW enviroment not set, please run cmsenv!"
   sys.exit()

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

#options = VarParsing.VarParsing ('standard')
options = VarParsing.VarParsing ()

options.register( 'usesLegacyProdNames',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Set to 1 if the reference files contains old (eg pfRecoTauProducer) PFTau product names"
                 ) 

options.register( 'scale',
                  'linear',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Set scale of yaxis on plots (linear/log)"
                  )

options.parseArguments()

process = cms.Process('MakingPlots')
process.DQMStore = cms.Service("DQMStore")

process.maxEvents = cms.untracked.PSet(            
    input = cms.untracked.int32(1)         
)

process.source = cms.Source("EmptySource")

# Get test file name
rootFilesInCurrentDirectory = glob.glob("*.root")
if len(rootFilesInCurrentDirectory) != 1:
   print "There must be one (and only one) root files in the current directory, otherwise I don't know what to compare!"
   sys.exit()
testRootFile = rootFilesInCurrentDirectory[0]
print "Loading test file: ", testRootFile

EventType = "unknown"
# Get event type
if testRootFile.find("ZTT"):
   EventType = 'ZTT'
elif testRootFile.find('QCD'):
   EventType = 'QCD'
elif testRootFile.find('ZEE'):
   EventType = 'ZEE'
elif testRootFile.find('ZMM'):
   EventType = 'ZMM'

# Output dir to hold the comparision
PlotOutputDir = "SummaryPlots"
if not os.path.exists(PlotOutputDir):
   os.makedirs(PlotOutputDir)

# Load plotting sequences
process.load("Validation.RecoTau.RecoTauValidation_cfi")
#set scale
from Validation.RecoTau.RecoTauValidation_cfi import SetLogScale
from Validation.RecoTau.RecoTauValidation_cfi import SetSmartLogScale
if options.scale == 'log':
   print "Setting everything to log scale"
   SetLogScale(process.plotTauValidation)

# Get helper functions
from Validation.RecoTau.RecoTauValidation_cfi import SetTestFileToPlot
from Validation.RecoTau.RecoTauValidation_cfi import SetReferenceFileToPlot
from Validation.RecoTau.RecoTauValidation_cfi import SetPlotDirectory
from Validation.RecoTau.RecoTauValidation_cfi import SetPlotOnlyStepByStep

#Set the test/reference files
SetTestFileToPlot(process, testRootFile)
SetReferenceFileToPlot(process, None)

# Set the right plot directory
SetPlotDirectory(process.plotTauValidation, PlotOutputDir)

# Only plot the summaries (we aren't comparing to anything)
SetPlotOnlyStepByStep(process.plotTauValidation)

#############################
# Path to be executed
###############################

process.p = cms.Path( process.loadTau 
                     +process.plotTauValidation
                     )

cfgLog = open(os.path.join(PlotOutputDir, "plot_config.py"), "write")

print >>cfgLog,  process.dumpPython()
 

