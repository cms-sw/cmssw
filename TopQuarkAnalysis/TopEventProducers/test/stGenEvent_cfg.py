import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/START36_V4-v1/0014/EEA7EEC1-FC49-DF11-9E91-003048678D9A.root'
    'rfio:///castor/cern.ch/user/s/snaumann/test/Spring10_SingleTop_sChannel-madgraph_AODSIM_START3X_V26_S09-v1.root',
    'rfio:///castor/cern.ch/user/s/snaumann/test/Spring10_SingleTop_tChannel-madgraph_AODSIM_START3X_V26_S09-v1.root',
    'rfio:///castor/cern.ch/user/s/snaumann/test/Spring10_SingleTop_tWChannel-madgraph_AODSIM_START3X_V26_S09-v1.root'
    )
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure genEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.stGenEvent_cff")

## path1
process.p1 = cms.Path(process.makeGenEvt)
