import FWCore.ParameterSet.Config as cms

#  Tracking Geometry
from SLHCUpgradeSimulations.Geometry.Phase1_cmsSimIdealGeometryXML_cfi import *
#from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
#from Geometry.CommonDetUnit.globalTrackingGeometryDB_cfi import *
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
es_prefer_TrackerEP = cms.ESPrefer("TrackerGeometricDetESModule", "trackerNumberingGeometry")
es_prefer_Trackerdigi = cms.ESPrefer("TrackerDigiGeometryESModule","trackerGeometry")

#hardwire these here
#trackerGeometry.trackerGeometryConstants = cms.PSet(
#    upgradeGeometry = cms.bool(True),
#    ROWS_PER_ROC = cms.int32(80),
#    COLS_PER_ROC = cms.int32(52),
#    BIG_PIX_PER_ROC_X = cms.int32(0),
#    BIG_PIX_PER_ROC_Y = cms.int32(0),
#    ROCS_X = cms.int32(2),
#    ROCS_Y = cms.int32(8)
#    )

#Tracker
#Configuration/ Geometry/ python/ GeometrySLHCReco_cff.py
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
trackerNumberingGeometry.fromDDD = cms.bool(True)
trackerNumberingGeometry.layerNumberPXB = cms.uint32(18)
trackerNumberingGeometry.totalBlade = cms.uint32(56)
trackerGeometry.applyAlignment = cms.bool(False)

#Muon
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.AlignedCaloGeometryDBReader_cfi import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
#from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometryDB_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometryDB_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometryDB_cff import *

