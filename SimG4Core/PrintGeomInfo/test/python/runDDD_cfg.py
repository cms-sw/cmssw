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
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    DumpSummary    = cms.untracked.bool(True),
    DumpLVTree     = cms.untracked.bool(True),
    DumpMaterial   = cms.untracked.bool(False),
    DumpLVList     = cms.untracked.bool(True),
    DumpLV         = cms.untracked.bool(False),
    DumpSolid      = cms.untracked.bool(True),
    DumpAttributes = cms.untracked.bool(False),
    DumpPV         = cms.untracked.bool(False),
    DumpRotation   = cms.untracked.bool(False),
    DumpReplica    = cms.untracked.bool(False),
    DumpTouch      = cms.untracked.bool(True),
    DumpSense      = cms.untracked.bool(True),
    DD4Hep         = cms.untracked.bool(False),
    Name           = cms.untracked.string('ME11*'),
    Names          = cms.untracked.vstring('EcalHitsEB'),
    type           = cms.string('PrintGeomInfoAction')
))
