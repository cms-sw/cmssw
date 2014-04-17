import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#Geometry
#
process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#Magnetic Field
#
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Detector simulation (Geant4-based)
#
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MaterialBudget = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('G4cerr','G4cout','MaterialBudget'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(14),
        MinEta = cms.double(-5.65),
        MaxEta = cms.double(-5.55),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(10.0),
        MaxE   = cms.double(10.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('matbdg_Castor.root')
)

process.p1 = cms.Path(process.generator*process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetHcal = cms.PSet(
        FillHisto    = cms.untracked.bool(False),
        PrintSummary = cms.untracked.bool(False),
        DoHCAL       = cms.untracked.bool(False),
        NBinPhi      = cms.untracked.int32(180),
        NBinEta      = cms.untracked.int32(100),
        EtaLow       = cms.untracked.double(-7.0),
        EtaHigh      = cms.untracked.double(-5.0),
        RMax         = cms.untracked.double(1.0),
        ZMax         = cms.untracked.double(18.0)
    ),
    type = cms.string('MaterialBudgetHcal')),
cms.PSet(
    CheckForHighEtPhotons = cms.untracked.bool(False),
    TrackMin     = cms.untracked.int32(0),
    TrackMax     = cms.untracked.int32(999999999),
    TrackStep    = cms.untracked.int32(1),
    EventMin     = cms.untracked.int32(0),
    EventMax     = cms.untracked.int32(0),
    EventStep    = cms.untracked.int32(1),
    PDGids       = cms.untracked.vint32(),
    VerboseLevel = cms.untracked.int32(0),
    G4Verbose    = cms.untracked.bool(True),
    DEBUG        = cms.untracked.bool(False),
    type      = cms.string('TrackingVerboseAction')
))


