import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append('ParticleListDrawer')

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_8_2/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0018/E8B5D618-96AF-DF11-835A-003048679070.root'
    #'/store/relval/CMSSW_3_8_2/RelValZEE/GEN-SIM-RECO/MC_38Y_V9-v1/0019/D85C639A-BEAF-DF11-8C04-0030486791C6.root'
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

## produce decaySubset
process.load("TopQuarkAnalysis.TopEventProducers.producers.TopDecaySubset_cfi")

## produce printout of particle listings (for debugging)
process.load("TopQuarkAnalysis.TopEventProducers.sequences.printGenParticles_cff")

## path
process.p = cms.Path(#process.printGenParticles *
                     process.decaySubset *
                     process.printDecaySubset)
