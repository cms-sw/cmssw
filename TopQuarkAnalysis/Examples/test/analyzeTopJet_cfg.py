import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# test cfg file for tqaflayer1 & 2 production from
# fullsim for semi-leptonic ttbar events 
#-------------------------------------------------
process = cms.Process("TEST")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.categories = cms.untracked.vstring('TEST')

#-------------------------------------------------
# process configuration
#-------------------------------------------------

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_1_X_2008-07-08_STARTUP_V4-AODSIM.100.root')
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

# Magnetic field now needs to be in the high-level py
process.load("Configuration.StandardSequences.MagneticField_cff")

#-------------------------------------------------
# tqaf configuration; if the TQAF Layer 1 is
# already in place yuo can comment the following
# two lines
#-------------------------------------------------

## std sequence for tqaf layer1
process.load("TopQuarkAnalysis.TopObjectProducers.tqafLayer1_full_cff")
process.p0 = cms.Path(process.tqafLayer1)

#-------------------------------------------------
# analyze jets
#-------------------------------------------------
from TopQuarkAnalysis.Examples.TopJetAnalyzer_cfi import analyzeJet
process.analyzeJet = analyzeJet

# register TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzeTopJet.root')
)

## end path   
process.p1 = cms.Path(process.analyzeJet)

