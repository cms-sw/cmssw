import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append('ParticleListDrawer')

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_3_8_2/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0018/E8B5D618-96AF-DF11-835A-003048679070.root'
    'rfio:///castor/cern.ch/user/s/snaumann/test/Spring10_SingleTop_sChannel-madgraph_AODSIM_START3X_V26_S09-v1.root',
    'rfio:///castor/cern.ch/user/s/snaumann/test/Spring10_SingleTop_tChannel-madgraph_AODSIM_START3X_V26_S09-v1.root',
    'rfio:///castor/cern.ch/user/s/snaumann/test/Spring10_SingleTop_tWChannel-madgraph_AODSIM_START3X_V26_S09-v1.root'
    )
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure genEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.stGenEvent_cff")

## produce printout of particle listings (for debugging)
process.load("TopQuarkAnalysis.TopEventProducers.sequences.printGenParticles_cff")

## path1
process.p1 = cms.Path(process.printGenParticles *
                      process.makeGenEvt *
                      process.printDecaySubset)
