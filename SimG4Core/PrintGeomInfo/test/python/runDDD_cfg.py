import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

#process.load('SimG4Core.PrintGeomInfo.testTotemGeometryXML_cfi')
#process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
#process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
#process.load('Geometry.EcalCommonData.ecalSimulationParameters_cff')
#process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cff')
process.load('Configuration.Geometry.GeometryExtended2026D41_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from SimG4Core.PrintGeomInfo.g4PrintGeomInfo_cfi import *

process = printGeomInfo(process)

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

process.g4SimHits.Watchers.Names = cms.untracked.vstring('HGCalEE')
#process.g4SimHits.Watchers.Names = cms.untracked.vstring('Internal_CSC_for_TotemT1_Plane_0_0_5', 'Internal_CSC_for_TotemT1_Plane_1_0_5','Internal_CSC_for_TotemT1_Plane_2_0_5','Internal_CSC_for_TotemT1_Plane_3_0_5','Internal_CSC_for_TotemT1_Plane_4_0_5','Internal_CSC_for_TotemT1_Plane_0_5_5','Internal_CSC_for_TotemT1_Plane_1_5_5','Internal_CSC_for_TotemT1_Plane_2_5_5','Internal_CSC_for_TotemT1_Plane_3_5_5','Internal_CSC_for_TotemT1_Plane_4_5_5','TotemT2gem_driftspace7r')
