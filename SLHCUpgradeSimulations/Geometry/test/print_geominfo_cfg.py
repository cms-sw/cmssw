import FWCore.ParameterSet.Config as cms

process = cms.Process("PrintGeom")

#process.load("SimG4Core.PrintGeomInfo.testTotemGeometryXML_cfi")
# choose the current geometry
#process.load("Geometry.CMSCommonData.trackerSimGeometryXML_cfi")
# or choose an upgrade geometry
process.load("SLHCUpgradeSimulations.Geometry.Phase1_R34F16_cmsSimIdealGeometryXML_cff")
#process.load("SLHCUpgradeSimulations.Geometry.hybrid_cmsIdealGeometryXML_cff")
#process.load("SLHCUpgradeSimulations.Geometry.longbarrel_cmsIdealGeometryXML_cff")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'MC_31X_V8::All'

process.TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
    fromDDD = cms.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('FwkJob'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    FrameworkJobReport = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

#process.load("Configuration.StandardSequences.Generator_cff")

process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
process.generator.PGunParameters.PartID[0] = 13
process.generator.PGunParameters.MinPt = 0.9
process.generator.PGunParameters.MaxPt = 50.0
process.generator.PGunParameters.MinEta = -2.4
process.generator.PGunParameters.MaxEta = 2.4
process.generator.AddAntiParticle = False

#process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
#    enableDivByZeroEx = cms.untracked.bool(False),
#    enableInvalidEx   = cms.untracked.bool(True),
#    enableOverFlowEx  = cms.untracked.bool(False),
#    enableUnderFlowEx = cms.untracked.bool(False)
#)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789),
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.load("SimG4Core.Application.g4SimHits_cfi")

process.p1 = cms.Path(process.generator*process.g4SimHits)
#process.p1 = cms.Path(process.g4SimHits)

process.g4SimHits.Physics.type            = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.UseMagneticField        = False
process.g4SimHits.Physics.DummyEMPhysics  = True
process.g4SimHits.Physics.DefaultCutValue = 10. 
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
	DumpSummary    = cms.untracked.bool(True),
	DumpLVTree     = cms.untracked.bool(False),
	DumpLVMatBudget= cms.untracked.bool(True),
# Standard geometry
#        LVNames2Dump   = cms.untracked.vstring('PixelBarrelLayer0','PixelBarrelLayer1','PixelBarrelLayer2'),
#        LVNames2Dump   = cms.untracked.vstring('PixelBarrel'),
#        LVNames2Dump   = cms.untracked.vstring('PixelForwardZPlus','PixelForwardZMinus'),
#        Radius2Use     = cms.untracked.vdouble(44.2558, 73.1232, 101.776),
#        Z2Use          = cms.untracked.vdouble(567.8, 567.8, 567.8),
#        LVNames2Dump   = cms.untracked.vstring('PixelBarrelLayer0','PixelBarrelLayer1','PixelBarrelLayer2',
#'PixelBarrelSuppTub', 'PixelBarrelSuppBox', 'PixelBarrelShield1', 'PixelBarrelShield2', 'PixelBarrelShield3', 'PixelBarrelShield4'),
#        Radius2Use     = cms.untracked.vdouble(44.2558, 73.1232, 101.776,
#101.776, 101.776, 44.2558, 44.2558, 101.776, 101.776),
#        Z2Use          = cms.untracked.vdouble(567.8, 567.8, 567.8,
#567.8, 567.8, 567.8, 567.8, 567.8, 567.8),
#
# Phase 1 geometry
        LVNames2Dump   = cms.untracked.vstring('PixelBarrelLayer0','PixelBarrelLayer1','PixelBarrelLayer2',
                                               'PixelBarrelLayer3','BEAM'),
        Radius2Use     = cms.untracked.vdouble(38.7995, 67.8557, 108.92, 159.958, 30.0),
        Z2Use          = cms.untracked.vdouble(567.8, 567.8, 567.8, 567.8, 567.8),
#        LVNames2Dump   = cms.untracked.vstring('PixelBarrel'),
#        Radius2Use     = cms.untracked.vdouble(38.7995),
#        Z2Use          = cms.untracked.vdouble(567.8),
#        LVNames2Dump   = cms.untracked.vstring('PixelForward'),
#        Radius2Use     = cms.untracked.vdouble(38.7995),
#        Z2Use          = cms.untracked.vdouble(567.8),
# Hybrid geometry
#        LVNames2Dump   = cms.untracked.vstring('PixelBarrelLayer0','PixelBarrelLayer1','PixelBarrelLayer2',
#                                               'PixelBarrelLayer3','PixelBarrelLayerStack0','PixelBarrelLayerStack1','BEAM',
#                                               'TOBLayer0','TOBLayer1','TOBLayer4','TOBLayer5'),
#        Radius2Use     = cms.untracked.vdouble(38.9527, 67.9102, 108.943, 159.99, 252.588, 351.876, 30.0,
#                                               509.188, 692.92, 865.77, 1080.65),
#        Z2Use          = cms.untracked.vdouble(567.8, 567.8, 567.8, 567.8, 3185.2, 4185.2, 567.8,
#                                               2168.86, 2168.86, 2168.86, 2168.86),
# Longbarrel
#        LVNames2Dump   = cms.untracked.vstring('PixelBarrelLayer0','PixelBarrelLayer1','PixelBarrelLayer2',
#                                               'PixelBarrelLayer3','PixelBarrelLayerStack0','PixelBarrelLayerStack1',
#                                               'PixelBarrelLayerStack2','PixelBarrelLayerStack3','PixelBarrelLayerStack4',
#                                               'PixelBarrelLayerStack5','PixelBarrelLayerStack6','PixelBarrelLayerStack7',
#                                               'PixelBarrelLayerStack8','PixelBarrelLayerStack9'),
#        Radius2Use     = cms.untracked.vdouble(38.9544, 67.9106, 108.943, 159.990, 321.882, 361.684,
#                                               481.283, 521.189, 643.973, 683.920, 803.791, 843.756, 
#                                               985.655, 1025.63),
#        Z2Use          = cms.untracked.vdouble( 567.8,  567.8,  567.8,  567.8, 4185.2, 4185.2,
#                                               5385.2, 5385.2, 1196.0, 1196.0, 1196.0, 1196.0,
#                                               5385.2, 5385.2),
	DumpMaterial   = cms.untracked.bool(False),
	DumpLVList     = cms.untracked.bool(False),
	DumpLV         = cms.untracked.bool(False),
	DumpSolid      = cms.untracked.bool(False),
	DumpAttributes = cms.untracked.bool(False),
	DumpPV         = cms.untracked.bool(False),
	DumpRotation   = cms.untracked.bool(False),
	DumpReplica    = cms.untracked.bool(False),
	DumpTouch      = cms.untracked.bool(False),
	DumpSense      = cms.untracked.bool(False),
	Name           = cms.untracked.string('PixelBarrel*'),
	Names          = cms.untracked.vstring('PixelBarrelActiveFull'),
	type           = cms.string('PrintGeomMatInfo')
))

