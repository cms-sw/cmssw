import FWCore.ParameterSet.Config as cms

validationMuonRPCDigis = cms.EDAnalyzer("RPCDigiValid",
    # Label to retrieve Digis from the event - not used anymore, switched to tags
    # digiLabel = cms.untracked.string('simMuonRPCDigis'),

    # Tag for Digis event data retrieval
    rpcDigiTag = cms.untracked.InputTag("simMuonRPCDigis"),
    # Tag for simulated hits event data retrieval
    simHitTag = cms.untracked.InputTag("g4SimHits", "MuonRPCHits"),

    # Name of the root file which will contain the histos
    outputFile = cms.untracked.string('')
)


