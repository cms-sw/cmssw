import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Run2_cff import Run2
#process = cms.Process('G4PrintGeometry',Run2)
#process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process('G4PrintGeometry',Run3_DDD)
process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')

#from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
#process = cms.Process('G4PrintGeometry',Phase2C11)
#process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')

from SimG4Core.PrintGeomInfo.g4PrintGeomInfo_cfi import *

process = printGeomInfo(process)

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    DumpSummary      = cms.untracked.bool(True),
    DumpLVTree       = cms.untracked.bool(False),
    DumpMaterial     = cms.untracked.bool(False),
    DumpLVList       = cms.untracked.bool(False),
    DumpLV           = cms.untracked.bool(False),
    DumpSolid        = cms.untracked.bool(False),
    DumpAttributes   = cms.untracked.bool(False),
    DumpPV           = cms.untracked.bool(False),
    DumpRotation     = cms.untracked.bool(False),
    DumpReplica      = cms.untracked.bool(False),
    DumpTouch        = cms.untracked.bool(False),
    DumpSense        = cms.untracked.bool(False),
    DumpRegion       = cms.untracked.bool(False),
    DD4hep           = cms.untracked.bool(False),
    Name             = cms.untracked.string(''),
    Names            = cms.untracked.vstring(''),
    MaterialFileName = cms.untracked.string('matfileDDD.txt'),
    SolidFileName    = cms.untracked.string('solidfileDDD.txt'),
    LVFileName       = cms.untracked.string('lvfileDDD.txt'),
    PVFileName       = cms.untracked.string('pvfileDDD.txt'),
    TouchFileName    = cms.untracked.string('touchfileDDD.txt'),
    RegionFileName   = cms.untracked.string('regionfileDDD.txt'),
    FileDetail       = cms.untracked.bool(True),
    type             = cms.string('PrintGeomInfoAction')
))
