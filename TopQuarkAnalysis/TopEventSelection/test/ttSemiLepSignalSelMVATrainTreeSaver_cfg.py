import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_6_0/RelValTTbar/GEN-SIM-RECO/START36_V4-v1/0014/EEA7EEC1-FC49-DF11-9E91-003048678D9A.root'
     ),
     skipEvents = cms.untracked.uint32(0)
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry & conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START38_V7::All')

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## std sequence for ttGenEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

## configure mva trainer
process.load("TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVATrainTreeSaver_cff")

## make trainer looper known to the process
from TopQuarkAnalysis.TopEventSelection.TtSemiLepSignalSelMVATrainTreeSaver_cff import looper
process.looper = looper

## produce pat objects and ttGenEvt and make mva training
process.p = cms.Path(process.patDefaultSequence *
                     process.makeGenEvt *
                     process.saveTrainTree)
