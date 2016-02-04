import FWCore.ParameterSet.Config as cms

rpcRecHitV = cms.EDAnalyzer("RPCRecHitValid",
    subDir = cms.string("RPC/RPCRecHitV/SimVsReco"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    recHit = cms.InputTag("rpcRecHits"),
    standAloneMode = cms.untracked.bool(False),
    rootFileName = cms.untracked.string("")
)

rpcRecHitValidation_step = cms.Sequence(rpcRecHitV)
