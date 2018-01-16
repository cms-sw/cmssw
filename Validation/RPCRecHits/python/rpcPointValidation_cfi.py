import FWCore.ParameterSet.Config as cms

dtVsRPCRecHitV = DQMStep1Module('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/DTVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

cscVsRPCRecHitV = DQMStep1Module('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/CSCVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

trackVsRPCRecHitV = DQMStep1Module('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/TrackVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

simVsDTExtV = DQMStep1Module('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsDTExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
)

simVsCSCExtV = DQMStep1Module('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsCSCExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
)

simVsTrackExtV = DQMStep1Module('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsTrackExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
)

rpcPointVsRecHitValidation_step = cms.Sequence(dtVsRPCRecHitV+cscVsRPCRecHitV)#+trackVsRPCRecHitV)
simVsRPCPointValidation_step = cms.Sequence(simVsDTExtV+simVsCSCExtV)#+simVsTrackExtV)

