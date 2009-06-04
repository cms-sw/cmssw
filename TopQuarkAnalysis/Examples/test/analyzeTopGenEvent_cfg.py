import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# test cfg file for tqaflayer1 & 2 production from
# fullsim for semi-leptonic ttbar events 
#-------------------------------------------------
process = cms.Process("TEST")

## configure message logger
## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
## dump content of TopGenEvent
process.MessageLogger.categories.append('TopGenEvent:dump')
## print final pruned listing of top decay chain
process.MessageLogger.categories.append('TopGenEventAnalyzer::selection')
process.MessageLogger.cout = cms.untracked.PSet(
 INFO = cms.untracked.PSet(
   limit = cms.untracked.int32(10),
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
   ),
   #skipEvents = cms.untracked.uint32(300)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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

## std sequence for tqaf layer2 ttGenEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
## process.decaySubset.addRadiatedGluons = False
process.load("TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi")
process.ttDecaySelection.src = "genEvt"
process.ttDecaySelection.allowedTopDecays.decayBranchA.electron = True
process.ttDecaySelection.allowedTopDecays.decayBranchA.muon     = True
process.ttDecaySelection.allowedTopDecays.decayBranchB.electron = True
process.ttDecaySelection.allowedTopDecays.decayBranchB.muon     = True
process.ttDecaySelection.allowedTauDecays.leptonic   = False
process.ttDecaySelection.allowedTauDecays.oneProng   = False
process.ttDecaySelection.allowedTauDecays.threeProng = False
## process.p1 = cms.Path(process.makeGenEvt * process.ttDecaySelection)

#-------------------------------------------------
# analyze genEvent
#-------------------------------------------------
from TopQuarkAnalysis.Examples.TopGenEventAnalyzer_cfi import analyzeTopGenEvent
process.analyzeTopGenEvent = analyzeTopGenEvent

# register TFService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzeTopGenEvent.root')
)

## analysis path   
process.p2 = cms.Path(process.makeGenEvt *
                      #process.ttDecaySelection *
                      process.analyzeTopGenEvent
                      )
