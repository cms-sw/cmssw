import FWCore.ParameterSet.Config as cms

#service = DaqMonitorROOTBackEnd{}
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
#include "RecoTracker/GeometryESProducer/data/TrackereRecoGeometryESProducer.cfi"
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from SimTracker.TrackAssociation.TrackAssociatorByPosition_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from Validation.RecoMuon.GlobalMuonMatchAnalyzer_cfi import *
TrackAssociatorByPosition.method = 'dist'
TrackAssociatorByPosition.MinIfNoMatch = True
TrackAssociatorByPosition.QCut = 10.

