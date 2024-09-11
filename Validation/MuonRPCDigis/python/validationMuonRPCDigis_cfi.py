import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
validationMuonRPCDigis = DQMEDAnalyzer('RPCDigiValid',

    # Tag for Digis event data retrieval
    rpcDigiTag = cms.untracked.InputTag("simMuonRPCDigis"),
    # Tag for simulated hits event data retrieval
    simHitTag = cms.untracked.InputTag("g4SimHits", "MuonRPCHits"),

    # Flag to turn on/off timing plots
    digiTime = cms.untracked.bool(False)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(validationMuonRPCDigis, simHitTag = "MuonSimHits:MuonRPCHits")

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(validationMuonRPCDigis, digiTime = True)
