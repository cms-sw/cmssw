import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# test cfg file for tqaflayer1 & 2 production from
# fullsim for semi-leptonic ttbar events 
#-------------------------------------------------
process = cms.Process("TEST")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
## show JEC from pat jet
process.MessageLogger.categories.append('TopJetAnalyzer_jec')
## show JetCorrFactors from pat jet
process.MessageLogger.categories.append('JetCorrFactors')
## show JetCorrFactorsProducer from pat jet
process.MessageLogger.categories.append('JetCorrFactorsProducer')
process.MessageLogger.cout = cms.untracked.PSet(
 INFO = cms.untracked.PSet(
   limit = cms.untracked.int32(0),
  )
)

#-------------------------------------------------
# process configuration
#-------------------------------------------------

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
   # PAT test sample for 2.2.X
    'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_2_X_2008-11-03-STARTUP_V7-AODSIM.100.root'
    )
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")

## configure conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V7::All')

# Magnetic field now needs to be in the high-level py
process.load("Configuration.StandardSequences.MagneticField_cff")


#-------------------------------------------------
# tqaf configuration; if the TQAF Layer 1 is
# already in place you can comment the following
# two lines
#-------------------------------------------------

## to apply jet correction factors different from
## the pat default uncomment this; needs to be
## called bevor pat/tqafLayer1 is produced
## process.load("TopQuarkAnalysis.TopObjectProducers.tools.switchJetCorrections_cff")

## std sequence for tqaf layer1
process.load("TopQuarkAnalysis.TopObjectProducers.tqafLayer1_cff")

## necessary fixes to run 2.2.X on 2.1.X data
## comment this when running on samples produced
## with 22X
## from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run22XonSummer08AODSIM
## run22XonSummer08AODSIM(process)

#-------------------------------------------------
# process paths;
#-------------------------------------------------

process.p0   = cms.Path(process.tqafLayer1)

#-------------------------------------------------
# analyze jets
#-------------------------------------------------
from TopQuarkAnalysis.Examples.TopJetAnalyzer_cfi import analyzeJet
process.analyzeJet = analyzeJet

process.jetCorrFactors.sampleType = 0

# register TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzeTopJet.root')
)

## end path   
process.p1 = cms.Path(process.analyzeJet)

