import FWCore.ParameterSet.Config as cms

from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
DQMStore = cms.Service("DQMStore")

muonIdVal = cms.EDFilter("MuonIdVal",
    inputCSCSegmentCollection = cms.InputTag("cscSegments"),
    inputMuonCollection = cms.InputTag("muons"),
    outputFile = cms.string('muonIdValPlots.root'),
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputTrackCollection = cms.InputTag("generalTracks")
)


