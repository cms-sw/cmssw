import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtVsRPCRecHitV = DQMEDAnalyzer('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/DTVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
cscVsRPCRecHitV = DQMEDAnalyzer('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/CSCVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
trackVsRPCRecHitV = DQMEDAnalyzer('RPCPointVsRecHit',
    subDir = cms.string("RPC/RPCRecHitV/TrackVsReco"),
    refHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
    recHit = cms.InputTag("rpcRecHits"),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
simVsDTExtV = DQMEDAnalyzer('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsDTExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCDTExtrapolatedPoints"),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
simVsCSCExtV = DQMEDAnalyzer('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsCSCExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCCSCExtrapolatedPoints"),
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
simVsTrackExtV = DQMEDAnalyzer('RPCRecHitValid',
    subDir = cms.string("RPC/RPCRecHitV/SimVsTrackExt"),
    simHit = cms.InputTag("g4SimHits", "MuonRPCHits"),
    simTrack = cms.InputTag("mix", "MergedTrackTruth"),
    muon = cms.InputTag("muons"),
    recHit = cms.InputTag("rpcPointProducer", "RPCTrackExtrapolatedPoints"),
)

rpcPointVsRecHitValidation_step = cms.Sequence(dtVsRPCRecHitV+cscVsRPCRecHitV)#+trackVsRPCRecHitV)
simVsRPCPointValidation_step = cms.Sequence(simVsDTExtV+simVsCSCExtV)#+simVsTrackExtV)

