import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 1

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(relValTTbar)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

## configure geometry & conditions
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## std sequence for ttGenEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

## filter for full-hadronic 
process.load("TopQuarkAnalysis.TopSkimming.ttDecayChannelFilters_cff")

## configure mva trainer
process.load("TopQuarkAnalysis.TopEventSelection.TtFullHadSignalSelMVATrainTreeSaver_cff")

## make trainer looper known to the process
from TopQuarkAnalysis.TopEventSelection.TtFullHadSignalSelMVATrainTreeSaver_cff import looper
process.looper = looper

## to tell the MVA trainer that an event is background and not signal
#process.buildTraintree.whatData = 0

## to filter ttbar background events instead of fully hadronic signal events
#process.ttFullHadronicFilter.invert = True

## jet count filter
process.load("PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi")

## setup jet collection, right now at least 6 jets needed for the MVA trainer/computer
process.leadingJetSelection = process.countPatJets.clone(src = 'selectedPatJets',
                                                         minNumber = 6
                                                         )

## produce pat objects and ttGenEvt and make mva training
process.p = cms.Path(process.ttFullHadronicFilter *
                     process.patDefaultSequence *
                     process.leadingJetSelection *
                     process.saveTrainTree)
