import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtVsRPCRecHitV = DQMEDAnalyzer('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/DTVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

cscVsRPCRecHitV = DQMEDAnalyzer('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/CSCVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

trackVsRPCRecHitV = DQMEDAnalyzer('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/TrackVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

simVsDTExtV = DQMEDAnalyzer('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsDTExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
)

simVsCSCExtV = DQMEDAnalyzer('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsCSCExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
)

simVsTrackExtV = DQMEDAnalyzer('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsTrackExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
)

rpcPointVsRecHitValidation_step = cms.Sequence(dtVsRPCRecHitV+cscVsRPCRecHitV)#+trackVsRPCRecHitV)
simVsRPCPointValidation_step = cms.Sequence(simVsDTExtV+simVsCSCExtV)#+simVsTrackExtV)

