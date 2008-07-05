import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#Geometry
#
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#Magnetic Field
#
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Detector simulation (Geant4-based)
#
process.load("SimG4Core.Application.g4SimHits_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876)
    ),
    sourceSeed = cms.untracked.uint32(135799753)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MaterialBudget = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('MaterialBudget'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(14),
        MaxEta = cms.untracked.double(3.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(2.65),
        MinE = cms.untracked.double(9.99),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxE = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetHcal = cms.PSet(
        NbinEta = cms.untracked.int32(260),
        etaLow = cms.untracked.double(-3.0),
        etaHigh = cms.untracked.double(3.0),
        ZMax = cms.untracked.double(14.0),
        MaxEta = cms.untracked.double(5.2),
        RMax = cms.untracked.double(5.0),
        NbinPhi = cms.untracked.int32(180),
        HistoFile = cms.untracked.string('matbdg_HCAL.root')
    ),
    type = cms.string('MaterialBudgetHcal')
))


