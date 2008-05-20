import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
MuonTrackAnalysisParameters = cms.PSet(
    MuonUpdatorAtVertex,
    MuonServiceProxy,
    CSCSimHit = cms.InputTag("MuonCSCHits","g4SimHits"),
    DTSimHit = cms.InputTag("MuonDTHits","g4SimHits"),
    EtaRange = cms.int32(0), ## all=0,barrel=1,endacap=2

    RPCSimHit = cms.InputTag("MuonRPCHits","g4SimHits")
)
STAMuonAnalyzer = cms.EDAnalyzer("MuonTrackAnalyzer",
    MuonTrackAnalysisParameters,
    DoSeedsAnalysis = cms.untracked.bool(True),
    rootFileName = cms.untracked.string('STA.root'),
    Tracks = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    #    untracked bool DoTracksAnalysis = false
    MuonSeed = cms.InputTag("MuonSeed")
)

GLBMuonAnalyzer = cms.EDAnalyzer("MuonTrackAnalyzer",
    MuonTrackAnalysisParameters,
    Tracks = cms.InputTag("globalMuons"),
    rootFileName = cms.untracked.string('GLB.root')
)

TrackerMuonAnalyzer = cms.EDAnalyzer("MuonTrackAnalyzer",
    MuonTrackAnalysisParameters,
    Tracks = cms.InputTag("ctfWithMaterialTracks"),
    rootFileName = cms.untracked.string('Tracker.root')
)

MuonTrackResidualAnalyzer = cms.EDAnalyzer("MuonTrackResidualAnalyzer",
    MuonTrackAnalysisParameters,
    MuonTrack = cms.InputTag("standAloneMuons"),
    rootFileName = cms.untracked.string('ResidualSTA.root')
)



