import FWCore.ParameterSet.Config as cms

process = cms.Process("PrintGeom")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(0)
)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(13),
        MinEta = cms.untracked.double(-2.5),
        MaxEta = cms.untracked.double(2.5),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinPt  = cms.untracked.double(9.99),
        MaxPt  = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)


process.EnableFloatingPointExceptions = cms.Service("EnableFloatingPointExceptions",
    enableDivByZeroEx = cms.untracked.bool(False),
    enableInvalidEx   = cms.untracked.bool(True),
    enableOverFlowEx  = cms.untracked.bool(False),
    enableUnderFlowEx = cms.untracked.bool(False)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        VtxSmeared = cms.untracked.uint32(98765432)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.load("SimG4Core.Application.g4SimHits_cfi")

process.p1 = cms.Path(process.g4SimHits)

process.g4SimHits.UseMagneticField        = False
process.g4SimHits.Physics.DefaultCutValue = 10. 
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
	Name           = cms.untracked.string('HCal*'),
	type           = cms.string('PrintSensitive')
))
