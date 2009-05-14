import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TtSemiLeptonicEvent')
process.MessageLogger.categories.append('TtSemiLepKinFitter')
#process.MessageLogger.categories.append('KinFitter')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default             = cms.untracked.PSet( limit = cms.untracked.int32( 0) ),
    TtSemiLeptonicEvent = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    TtSemiLepKinFitter  = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
#   KinFitter           = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_2_X_2008-11-03-STARTUP_V7-AODSIM.100.root'
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
process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')

## load magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## std sequence to produce the ttGenEvt
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

## std sequence to produce the ttSemiLepEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff")
process.ttSemiLepEvent.verbosity = 1

## change maximum number of jets taken into account per event (default: 4)
## process.ttSemiLepEvent.maxNJets = 5

## change maximum number of jet combinations taken into account per event (default: 1)
## process.findTtSemiLepJetCombMVA.maxNComb        = -1
## process.kinFitTtSemiLepEventHypothesis.maxNComb = -1

## default replacements from B22X -> B22X_v2 in other packages
process.ttSemiLepHypGeom.mets               = "layer1METs"
process.ttSemiLepHypKinFit.mets             = "layer1METs"
process.ttSemiLepHypMVADisc.mets            = "layer1METs"
process.ttSemiLepHypGenMatch.mets           = "layer1METs"
process.findTtSemiLepJetCombMVA.mets        = "layer1METs"
process.ttSemiLepHypMaxSumPtWMass.mets      = "layer1METs"
process.ttSemiLepHypWMassMaxSumPt.mets      = "layer1METs"
process.kinFitTtSemiLepEventHypothesis.mets = "layer1METs"

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
