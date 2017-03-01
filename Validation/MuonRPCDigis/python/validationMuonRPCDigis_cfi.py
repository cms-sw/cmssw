import FWCore.ParameterSet.Config as cms

validationMuonRPCDigis = cms.EDAnalyzer("RPCDigiValid",

    # Tag for Digis event data retrieval
    rpcDigiTag = cms.untracked.InputTag("simMuonRPCDigis"),
    # Tag for simulated hits event data retrieval
    simHitTag = cms.untracked.InputTag("g4SimHits", "MuonRPCHits"),

    # Name of the root file which will contain the histos
    outputFile = cms.untracked.string('')
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    validationMuonRPCDigis.simHitTag = cms.untracked.InputTag("MuonSimHits","MuonRPCHits")
    
