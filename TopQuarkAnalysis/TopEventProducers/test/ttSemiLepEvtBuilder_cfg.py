import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TtSemiLepKinFitter')
process.MessageLogger.categories.append('TtSemiLeptonicEvent')
process.MessageLogger.cerr.TtSemiLeptonicEvent = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_3_0_pre4/RelValTTbar/GEN-SIM-RECO/MC_31X_V8-v1/0000/FCC5D1DF-C3A6-DE11-81AA-001D09F28EA3.root'
    )
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")

## configure conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V8::All')

## load magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## std sequence to produce the ttGenEvt
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

## std sequence to produce the ttSemiLepEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff")
process.ttSemiLepEvent.verbosity = 1

## choose which hypotheses to produce
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff import *
addTtSemiLepHypotheses(process,
                       ["kGeom", "kWMassMaxSumPt", "kMaxSumPtWMass", "kMVADisc", "kKinFit"]
                       )
#removeTtSemiLepHypGenMatch(process)

## change maximum number of jets taken into account per event (default: 4)
## process.ttSemiLepEvent.maxNJets = 5

## change maximum number of jet combinations taken into account per event (default: 1)
## process.findTtSemiLepJetCombMVA.maxNComb        = -1
## process.kinFitTtSemiLepEventHypothesis.maxNComb = -1

## process path
process.p = cms.Path(process.patDefaultSequence *
                     process.makeGenEvt         *
                     process.makeTtSemiLepEvent
                     )

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    fileName     = cms.untracked.string('ttSemiLepEvtBuilder.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring('drop *'),                      
    dropMetaDataForDroppedData = cms.untracked.bool(True)
)
process.outpath = cms.EndPath(process.out)

## pat content
from PhysicsTools.PatAlgos.patEventContent_cff import *
process.out.outputCommands += patTriggerEventContent
process.out.outputCommands += patExtraAodEventContent
process.out.outputCommands += patEventContentNoLayer1Cleaning
## tqaf content
from TopQuarkAnalysis.TopEventProducers.tqafEventContent_cff import *
process.out.outputCommands += tqafEventContent
