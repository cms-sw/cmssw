import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TtSemiLepHitFit')
process.MessageLogger.categories.append('HitFit')
process.MessageLogger.cerr.TtSemiLepHitFit = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.cerr.HitFit = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

## define input
from TopQuarkAnalysis.TopEventProducers.tqafInputFiles_cff import relValTTbar
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(relValTTbar)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)

)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry & conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#from Configuration.AlCa.autoCond import autoCond
from Configuration.AlCa.autoCond import autoCond 
process.GlobalTag.globaltag = autoCond['mc']

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## the resolutions in HitFit are currently only provided for |eta|<3!
## an object collection with appropriate cuts has to used (the module will run into exceptions otherwise)
process.selectedPatJets.cut = 'abs(eta) < 3.0'

## std sequence to produce the kinematic fit for semi-leptonic events
process.load("TopQuarkAnalysis.TopHitFit.TtSemiLepHitFitProducer_Muons_cfi")

## process path
process.p = cms.Path(process.patDefaultSequence *
                     process.hitFitTtSemiLepEvent
                     )

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents   = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),                               
    fileName = cms.untracked.string('ttSemiLepHitFitProducer.root'),
    outputCommands = cms.untracked.vstring('drop *')
)
process.out.outputCommands += ['keep *_hitFitTtSemiLepEvent_*_*']

## output path
process.outpath = cms.EndPath(process.out)
