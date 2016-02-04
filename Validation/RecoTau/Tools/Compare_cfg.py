import os
import sys
import shutil

try:
   ReleaseBase = os.path.join(os.environ['CMSSW_BASE'], "src")
   ReleaseVersion = os.environ['CMSSW_VERSION']
except KeyError:
   print "CMSSW enviroment not set, please run cmsenv!"
   sys.exit()

import glob
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

#options = VarParsing.VarParsing ('standard')
options = VarParsing.VarParsing ()

options.register( 'compareTo',
                  '',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Specify path to directory to compare to. e.g. Validation_CMSSW_2_2_9/ZTT_recoFiles"
                 )

options.register( 'testLabel',
                  ReleaseVersion,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Label for test release (this one)"
                 )

options.register( 'referenceLabel',
                  'NOLABEL',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Label for reference release (to compare too)"
                 )

options.register( 'referenceUsesLegacyProdNames',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Set to 1 if the reference files contains old (eg pfRecoTauProducer) PFTau product names"
                 ) 

options.register( 'usesLegacyProdNames',
                  0,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Set to 2 if ALL files contains old (eg pfRecoTauProducer) PFTau product names"
                 ) 

options.register( 'scale',
                  'linear',
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Set scale of yaxis on plots (linear/log/smartlog) smartlog option sets only high-purity (TaNC, electron, muon) discriminators to log"
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

#Get the reference file name
refDirectory = os.path.abspath(options.compareTo)
rootFilesInRefDirectory = glob.glob(os.path.join(refDirectory, "*.root"))
if len(rootFilesInCurrentDirectory) != 1:
   print "There must be one (and only one) root files in the test directory, otherwise I don't know what to compare!"
   sys.exit()

refRootFile = rootFilesInRefDirectory[0]
print "Loading reference file: ", refRootFile



CleanReferenceLabel = options.referenceLabel.replace(" ","").replace("-","")
CleanTestLabel      = options.testLabel.replace(" ","").replace("-","")

# Output dir to hold the comparision
PlotOutputDir = "ComparedTo" + CleanReferenceLabel

#Save the reference root file and configuration for safekeeping
RefOutputDir = os.path.join(PlotOutputDir, "ReferenceData")
if not os.path.exists(RefOutputDir):
   os.makedirs(RefOutputDir)

shutil.copy(refRootFile, RefOutputDir)
shutil.copytree(os.path.join(refDirectory, "Config"), os.path.join(RefOutputDir, "Config"))

PlotOutputDir = os.path.join(PlotOutputDir, "Plots")
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
elif options.scale == 'smartlog':
   print "Setting high purity discriminators to log scale"
   SetSmartLogScale(process.plotTauValidation)


# Get helper functions
from Validation.RecoTau.RecoTauValidation_cfi import SetTestFileToPlot
from Validation.RecoTau.RecoTauValidation_cfi import SetReferenceFileToPlot
from Validation.RecoTau.RecoTauValidation_cfi import SetPlotDirectory
from Validation.RecoTau.RecoTauValidation_cfi import SetTestAndReferenceLabels

#Set the test/reference files
SetTestFileToPlot(process, testRootFile)
SetReferenceFileToPlot(process, refRootFile)

# Switch to 22X style names if desired
from Validation.RecoTau.RecoTauValidation_cfi import SetCompareToLegacyProductNames
if options.usesLegacyProdNames == 1:
   from Validation.RecoTau.RecoTauValidation_cfi import UseLegacyProductNames
   # remove the tanc from the sequence
   process.plotTauValidation = cms.Sequence(process.plotTauValidationNoTanc)
   UseLegacyProductNames(process.plotTauValidation)
elif options.referenceUsesLegacyProdNames == 1:
   SetCompareToLegacyProductNames(process.plotTauValidation)

# Set the right plot directoy
print process.plotTauValidation
SetPlotDirectory(process.plotTauValidation, PlotOutputDir)

# Now the directories have been built (by SetPlotDirectory)
#  lets generate the webpages
for baseDir, producerPlotDir in [os.path.split(x) for x in filter(os.path.isdir, glob.glob(os.path.join(PlotOutputDir, "*")))]:
   baseDirCommand = "cd %s;" % baseDir
   webpageMaker   = "$VALTOOLS/make_comparison_webpage ";
   webpageOptions = "%s %s %s %s" % (CleanTestLabel, CleanReferenceLabel, producerPlotDir, EventType)
   os.system(baseDirCommand+webpageMaker+webpageOptions)



SetTestAndReferenceLabels(process.plotTauValidation, options.testLabel, options.referenceLabel)

#############################
# Path to be executed
###############################

process.p = cms.Path( process.loadTau 
                     +process.plotTauValidation
                     )

cfgLog = open(os.path.join(PlotOutputDir, "plot_config.py"), "write")

print >>cfgLog,  process.dumpPython()
 

