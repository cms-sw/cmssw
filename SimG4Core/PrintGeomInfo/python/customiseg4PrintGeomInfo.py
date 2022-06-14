import FWCore.ParameterSet.Config as cms
def customise(process):

    process.g4SimHits.UseMagneticField = False
    process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
    process.g4SimHits.Physics.DummyEMPhysics = True
    process.g4SimHits.Physics.DefaultCutValue = 10.

    process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
        DumpSummary    = cms.untracked.bool(True),
        DumpLVTree     = cms.untracked.bool(True),
        DumpMaterial   = cms.untracked.bool(False),
        DumpLVList     = cms.untracked.bool(True),
        DumpLV         = cms.untracked.bool(True),
        DumpSolid      = cms.untracked.bool(True),
        DumpAttributes = cms.untracked.bool(False),
        DumpPV         = cms.untracked.bool(True),
        DumpRotation   = cms.untracked.bool(False),
        DumpReplica    = cms.untracked.bool(False),
        DumpTouch      = cms.untracked.bool(False),
        DumpSense      = cms.untracked.bool(False),
        DumpParams     = cms.untracked.bool(False),
        DD4hep         = cms.untracked.bool(False),
        Name           = cms.untracked.string('CMS*'),
        Names          = cms.untracked.vstring(' '),
        type           = cms.string('PrintGeomInfoAction')
    ))

    if hasattr(process,'MessageLogger'):
        process.MessageLogger.G4cerr = cms.untracked.PSet()
        process.MessageLogger.G4cout = cms.untracked.PSet()

    return(process)

def customiseg4PrintGeomInfo(process):
    return customise(process)
