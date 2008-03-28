import FWCore.ParameterSet.Config as cms

from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

muonIdVal = cms.EDFilter("MuonIdVal",
    inputCSCSegmentCollection = cms.InputTag("cscSegments"),
    inputMuonCollection = cms.InputTag("muons"),
    outputFile = cms.string('muonIdValPlots.root'),
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputTrackCollection = cms.InputTag("generalTracks")
)


