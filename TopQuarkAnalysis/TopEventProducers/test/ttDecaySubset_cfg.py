import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
## print original listing of gen particles
process.MessageLogger.categories.append('TopDecaySubset_printSource')
## print final pruned listing of top decay chain
process.MessageLogger.categories.append('TopDecaySubset_printTarget')
process.MessageLogger.cout = cms.untracked.PSet(
 INFO = cms.untracked.PSet(
   limit = cms.untracked.int32(0),
   TopDecaySubset_printSource = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
   TopDecaySubset_printTarget = cms.untracked.PSet( limit = cms.untracked.int32(10) )
  )
)

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
   'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_2_X_2008-11-03-STARTUP_V7-AODSIM.100.root'
    )
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## std sequence to produce the decaySubset
process.load("TopQuarkAnalysis.TopEventProducers.producers.TopDecaySubset_cfi")

## process path
process.p = cms.Path(process.decaySubset)
