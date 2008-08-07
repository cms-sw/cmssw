import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
MuonTrackAnalysisParameters = cms.PSet(
    MuonUpdatorAtVertex,
    MuonServiceProxy,
    CSCSimHit = cms.InputTag("MuonCSCHits","g4SimHits"),
    DTSimHit = cms.InputTag("MuonDTHits","g4SimHits"),
    EtaRange = cms.int32(0),
    DataType = cms.InputTag("RealData"),
    rootFileName = cms.untracked.string('validationPlots.root'),
    RPCSimHit = cms.InputTag("MuonRPCHits","g4SimHits")
)
L2MuonTrackAnalyzer = cms.EDAnalyzer("MuonTrackAnalyzer",
    MuonTrackAnalysisParameters,
    DoSeedsAnalysis = cms.untracked.bool(True),
    dirName = cms.untracked.string('RecoMuonV/TrackAnalyzer/'),
    Tracks = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    MuonSeed = cms.InputTag("MuonSeed")
)

L3MuonTrackAnalyzer = cms.EDAnalyzer("MuonTrackAnalyzer",
    MuonTrackAnalysisParameters,
    Tracks = cms.InputTag("hltL3Muons"),
    dirName = cms.untracked.string('RecoMuonV/TrackAnalyzer/')
)

TrackerMuonAnalyzer = cms.EDAnalyzer("MuonTrackAnalyzer",
    MuonTrackAnalysisParameters,
    Tracks = cms.InputTag("hltL3Muons","L2Seeded"),
    dirName = cms.untracked.string('RecoMuonV/TrackAnalyzer/')
)

MuonTrackResidualAnalyzer = cms.EDAnalyzer("MuonTrackResidualAnalyzer",
    MuonTrackAnalysisParameters,
    dirName = cms.untracked.string('RecoMuonV/TrackResidualAnalyzer/'),
    MuonSeed = cms.InputTag("MuonSeed"),
    MuonTrack = cms.InputTag("hltL2Muons")
)


