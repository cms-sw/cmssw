import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
## add categories
## process.MessageLogger.categories.append('TopDecaySubset_printSource')
process.MessageLogger.categories.append('TopDecaySubset_printTarget')
## configure cout
process.MessageLogger.cout = cms.untracked.PSet(
 INFO = cms.untracked.PSet(
   limit = cms.untracked.int32(0),
## TopDecaySubset_printSource       = cms.untracked.PSet( limit = cms.untracked.int32(10) ),
   TopDecaySubset_printTarget       = cms.untracked.PSet( limit = cms.untracked.int32(10) )
  )
)

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0002/50D4BADB-FA32-DE11-BA01-000423D98DC4.root'
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
process.GlobalTag.globaltag = cms.string('IDEAL_31X::All')

## produce decaySubset
process.load("TopQuarkAnalysis.TopEventProducers.producers.TopDecaySubset_cfi")
## path
process.p = cms.Path(process.decaySubset)
