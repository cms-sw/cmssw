import FWCore.ParameterSet.Config as cms

process = cms.Process("PrintMaterialBudget")

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Geometry.CMSCommonData.cmsExtendedGeometry2015XML_cfi')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')

process.MessageLogger.destinations = cms.untracked.vstring("MatBudget.txt")

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
	Name           = cms.untracked.string('TIDF'),
	type           = cms.string('PrintMaterialBudgetInfo')
))
