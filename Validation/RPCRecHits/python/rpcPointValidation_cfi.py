import FWCore.ParameterSet.Config as cms

dtVsRPCRecHitV = cms.EDAnalyzer("RPCPointVsRecHit",
    subDir = cms.string("RPC/RPCRecHitV/DTVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
    standAloneMode = cms.untracked.bool(False),
    rootFileName = cms.untracked.string("")
)

cscVsRPCRecHitV = cms.EDAnalyzer("RPCPointVsRecHit",
    subDir = cms.string("RPC/RPCRecHitV/CSCVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
    standAloneMode = cms.untracked.bool(False),
    rootFileName = cms.untracked.string("")
)

trackVsRPCRecHitV = cms.EDAnalyzer("RPCPointVsRecHit",
    subDir = cms.string("RPC/RPCRecHitV/TrackVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
    standAloneMode = cms.untracked.bool(False),
    rootFileName = cms.untracked.string("")
)

simVsDTExtV = cms.EDAnalyzer("RPCRecHitValid",
    subDir = cms.string("RPC/RPCRecHitV/SimVsDTExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mergedtruth", "MergedTrackTruth"),
    recHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
    standAloneMode = cms.untracked.bool(False),
    rootFileName = cms.untracked.string("")
)

simVsCSCExtV = cms.EDAnalyzer("RPCRecHitValid",
    subDir = cms.string("RPC/RPCRecHitV/SimVsCSCExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mergedtruth", "MergedTrackTruth"),
    recHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
    standAloneMode = cms.untracked.bool(False),
    rootFileName = cms.untracked.string("")
)

simVsTrackExtV = cms.EDAnalyzer("RPCRecHitValid",
    subDir = cms.string("RPC/RPCRecHitV/SimVsTrackExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mergedtruth", "MergedTrackTruth"),
    recHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
    standAloneMode = cms.untracked.bool(False),
    rootFileName = cms.untracked.string("")
)

rpcPointVsRecHitValidation_step = cms.Sequence(dtVsRPCRecHitV+cscVsRPCRecHitV)#+trackVsRPCRecHitV)
simVsRPCPointValidation_step = cms.Sequence(simVsDTExtV+simVsCSCExtV)#+simVsTrackExtV)

