import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append('TopDecaySubset_printTarget')
process.MessageLogger.cerr.TopDecaySubset_printTarget = cms.untracked.PSet(
    limit = cms.untracked.int32(10)
)

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0008/2C8CD8FE-B6B5-DE11-ACB8-001D09F2905B.root'
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

## configure geometry & conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_3XY_V10::All')

## produce decaySubset
process.load("TopQuarkAnalysis.TopEventProducers.producers.TopDecaySubset_cfi")
## path
process.p = cms.Path(process.decaySubset)
