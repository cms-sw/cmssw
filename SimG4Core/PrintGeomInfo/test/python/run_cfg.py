import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

process.load('SimG4Core.PrintGeomInfo.testTotemGeometryXML_cfi')
#process.load('Configuration.Geometry.GeometryExtended2026D41_cff')

from SimG4Core.PrintGeomInfo.g4PrintGeomInfo_cfi import *

process = printGeomInfo(process)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('G4cerr', 'G4cout'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
    )
)

#process.g4SimHits.Watchers.Names = cms.untracked.vstring('HGCalEE')
process.g4SimHits.Watchers.Names = cms.untracked.vstring('Internal_CSC_for_TotemT1_Plane_0_0_5', 'Internal_CSC_for_TotemT1_Plane_1_0_5','Internal_CSC_for_TotemT1_Plane_2_0_5','Internal_CSC_for_TotemT1_Plane_3_0_5','Internal_CSC_for_TotemT1_Plane_4_0_5','Internal_CSC_for_TotemT1_Plane_0_5_5','Internal_CSC_for_TotemT1_Plane_1_5_5','Internal_CSC_for_TotemT1_Plane_2_5_5','Internal_CSC_for_TotemT1_Plane_3_5_5','Internal_CSC_for_TotemT1_Plane_4_5_5','TotemT2gem_driftspace7r')
