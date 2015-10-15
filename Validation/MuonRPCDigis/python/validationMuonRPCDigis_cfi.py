import FWCore.ParameterSet.Config as cms

validationMuonRPCDigis = cms.EDAnalyzer("RPCDigiValid",

    # Tag for Digis event data retrieval
    rpcDigiTag = cms.untracked.InputTag("simMuonRPCDigis"),
    rpcDigiForPileup = cms.untracked.InputTag("hltMuonRPCDigis"),
    # Tag for simulated hits event data retrieval
    simHitTag = cms.untracked.InputTag("g4SimHits", "MuonRPCHits"),

    # Name of the root file which will contain the histos
    outputFile = cms.untracked.string('')
)



