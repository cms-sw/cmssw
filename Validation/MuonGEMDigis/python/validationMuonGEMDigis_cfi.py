import FWCore.ParameterSet.Config as cms

validationMuonGEMDigis = cms.EDAnalyzer("GEMDigiValid",
    # Tag for Digis event data retrieval
    gemDigiTag = cms.untracked.InputTag("simMuonGEMDigis"),
    # Tag for simulated hits event data retrieval
    simHitTag = cms.untracked.InputTag("g4SimHits", "MuonGEMHits"),
    outputFile = cms.untracked.string('')
)



