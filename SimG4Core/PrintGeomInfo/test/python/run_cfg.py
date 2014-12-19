import FWCore.ParameterSet.Config as cms

process = cms.Process("PrintGeom")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimG4Core.PrintGeomInfo.testTotemGeometryXML_cfi")

process.MessageLogger.destinations = cms.untracked.vstring("Totem.txt")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(13),
        MinEta = cms.double(-2.5),
        MaxEta = cms.double(2.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinPt  = cms.double(9.99),
        MaxPt  = cms.double(10.01)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    generator = cms.PSet(
         initialSeed = cms.untracked.uint32(123456789),
         engineName = cms.untracked.string('HepJamesRandom')
    ),
    g4SimHits = cms.PSet(
         initialSeed = cms.untracked.uint32(11),
         engineName = cms.untracked.string('HepJamesRandom')
    )
)

process.load("SimG4Core.Application.g4SimHits_cfi")

process.p1 = cms.Path(process.generator*process.g4SimHits)

process.g4SimHits.Physics.type            = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.UseMagneticField        = False
process.g4SimHits.Physics.DummyEMPhysics  = True
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
	Name           = cms.untracked.string('TotemT*'),
	Names          = cms.untracked.vstring('Internal_CSC_for_TotemT1_Plane_0_0_5', 'Internal_CSC_for_TotemT1_Plane_1_0_5','Internal_CSC_for_TotemT1_Plane_2_0_5','Internal_CSC_for_TotemT1_Plane_3_0_5','Internal_CSC_for_TotemT1_Plane_4_0_5','Internal_CSC_for_TotemT1_Plane_0_5_5','Internal_CSC_for_TotemT1_Plane_1_5_5','Internal_CSC_for_TotemT1_Plane_2_5_5','Internal_CSC_for_TotemT1_Plane_3_5_5','Internal_CSC_for_TotemT1_Plane_4_5_5','TotemT2gem_driftspace7r'),
	type           = cms.string('PrintGeomInfoAction')
))
