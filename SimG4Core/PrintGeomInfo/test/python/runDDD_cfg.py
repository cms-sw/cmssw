import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
process.load('Geometry.EcalCommonData.ecalSimulationParameters_cff')
process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cff')
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')


from SimG4Core.PrintGeomInfo.g4PrintGeomInfo_cfi import *

process = printGeomInfo(process)

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()



process.g4SimHits.g4GeometryDD4hepSource = cms.bool(False)
process.g4SimHits.Watchers.Names = cms.untracked.vstring('EcalSB')
process.g4SimHits.Watchers.DD4Hep = cms.untracked.bool(False)
