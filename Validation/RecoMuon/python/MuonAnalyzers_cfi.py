import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
MuonTrackAnalysisParameters = cms.PSet(
    MuonUpdatorAtVertex,
    MuonServiceProxy,
    CSCSimHit = cms.InputTag("MuonCSCHits","g4SimHits"),
    DTSimHit = cms.InputTag("MuonDTHits","g4SimHits"),
    EtaRange = cms.int32(0), ## all=0,barrel=1,endacap=2

    DataType = cms.InputTag("RealData"),
    rootFileName = cms.untracked.string('validationPlots.root'),
    RPCSimHit = cms.InputTag("MuonRPCHits","g4SimHits")
)
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
STAMuonAnalyzer = DQMEDAnalyzer('MuonTrackAnalyzer',
    MuonTrackAnalysisParameters,
    DoSeedsAnalysis = cms.untracked.bool(True),
    dirName = cms.untracked.string('Muons/RecoMuonV/TrackAnalyzer/'),
    Tracks = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    MuonSeed = cms.InputTag("MuonSeed")
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
GLBMuonAnalyzer = DQMEDAnalyzer('MuonTrackAnalyzer',
    MuonTrackAnalysisParameters,
    Tracks = cms.InputTag("globalMuons"),
    dirName = cms.untracked.string('Muons/RecoMuonV/TrackAnalyzer/')
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
TrackerMuonAnalyzer = DQMEDAnalyzer('MuonTrackAnalyzer',
    MuonTrackAnalysisParameters,
    Tracks = cms.InputTag("generalTracks"),
    dirName = cms.untracked.string('Muons/RecoMuonV/TrackAnalyzer/')
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
MuonTrackResidualAnalyzer = DQMEDAnalyzer('MuonTrackResidualAnalyzer',
    MuonTrackAnalysisParameters,
    MuonTrack = cms.InputTag("standAloneMuons"),
    dirName = cms.untracked.string('Muons/RecoMuonV/TrackResidualAnalyzer/'),
    MuonSeed = cms.InputTag("MuonSeed")
)



