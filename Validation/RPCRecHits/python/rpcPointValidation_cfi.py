import FWCore.ParameterSet.Config as cms

dtVsRPCRecHitV = cms.EDAnalyzer("RPCPointVsRecHit",
    subDir = cms.string("RPC/RPCRecHitV/DTVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

cscVsRPCRecHitV = cms.EDAnalyzer("RPCPointVsRecHit",
    subDir = cms.string("RPC/RPCRecHitV/CSCVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

trackVsRPCRecHitV = cms.EDAnalyzer("RPCPointVsRecHit",
    subDir = cms.string("RPC/RPCRecHitV/TrackVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

simVsDTExtV = cms.EDAnalyzer("RPCRecHitValid",
    subDir = cms.string("RPC/RPCRecHitV/SimVsDTExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
)

simVsCSCExtV = cms.EDAnalyzer("RPCRecHitValid",
    subDir = cms.string("RPC/RPCRecHitV/SimVsCSCExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
)

simVsTrackExtV = cms.EDAnalyzer("RPCRecHitValid",
    subDir = cms.string("RPC/RPCRecHitV/SimVsTrackExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
)

rpcPointVsRecHitValidation_step = cms.Sequence(dtVsRPCRecHitV+cscVsRPCRecHitV)#+trackVsRPCRecHitV)
simVsRPCPointValidation_step = cms.Sequence(simVsDTExtV+simVsCSCExtV)#+simVsTrackExtV)

