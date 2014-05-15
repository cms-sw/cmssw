import FWCore.ParameterSet.Config as cms

rpcRecHitV = cms.EDAnalyzer("RPCRecHitValid",
    subDir = cms.string("RPC/RPCRecHitV/SimVsReco"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    recHit = cms.InputTag("rpcRecHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    simHitAssoc = cms.InputTag("simHitTPAssocProducer"),
    muon = cms.InputTag("muons"),
)

rpcRecHitValidation_step = cms.Sequence(rpcRecHitV)
