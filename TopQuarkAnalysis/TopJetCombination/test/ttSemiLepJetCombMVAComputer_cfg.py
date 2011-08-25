import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

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
    wantSummary = cms.untracked.bool(False)
)

## configure geometry & conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## configure mva computer
process.load("TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombMVAComputer_cff")
## change maximum number of jets taken into account per event (default: 4)
#process.findTtSemiLepJetCombMVA.maxNJets = 5

## produce pat objects and perform MVA for jet-parton association
process.p = cms.Path(process.patDefaultSequence *
                     process.findTtSemiLepJetCombMVA)

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents   = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),                               
    fileName = cms.untracked.string('ttSemiLepJetCombMVAComputer_muons.root'),
    outputCommands = cms.untracked.vstring('drop *')
)
process.out.outputCommands += ['keep *_findTtSemiLepJetCombMVA_*_*']

## output path
process.outpath = cms.EndPath(process.out)
