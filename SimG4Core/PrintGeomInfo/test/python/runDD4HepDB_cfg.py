import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
process = cms.Process('G4PrintGeometry',Run3_dd4hep)
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '120X_mcRun3_2021_realistic_dd4hep_v1', '')

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
    DD4Hep           = cms.untracked.bool(True),
    Name             = cms.untracked.string(''),
    Names            = cms.untracked.vstring(''),
    MaterialFileName = cms.untracked.string('matfileDD4HepDB.txt'),
    SolidFileName    = cms.untracked.string('solidfileDD4HepDB.txt'),
    LVFileName       = cms.untracked.string('lvfileDD4HepDB.txt'),
    PVFileName       = cms.untracked.string('pvfileDD4HepDB.txt'),
    TouchFileName    = cms.untracked.string('touchfileDD4HepDB.txt'),
    FileDetail       = cms.untracked.bool(True),
    type             = cms.string('PrintGeomInfoAction')
))
