import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#Geometry
#
process.load("Geometry.CMSCommonData.cmsExtendedGeometryXML_cfi")
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
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('MaterialBudget'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MaterialBudget = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    )
)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(14),
        MinEta = cms.double(-7.0),
        MaxEta = cms.double(-4.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(10.0),
        MaxE   = cms.double(10.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('matbdg_Fwd.root')
)

process.p1 = cms.Path(process.generator*process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Generator.ApplyEtaCuts = False

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetForward = cms.PSet(
        DetectorTypes = cms.vstring('BeamPipe','Tracker','EM Calorimeter','Hadron Calorimeter','Forward Calorimeter','TOTEM','CASTOR','Forward Shield','Muon System'),
        Constituents  = cms.vint32(4,2,1,1,1,2,2,1,1),
        StackOrder    = cms.vint32(1,2,3,4,6,8,9,7,5),
        DetectorNames = cms.vstring('BEAM','BEAM1','BEAM2','BEAM3','Tracker','PLT','ECAL','CALO','VCAL','TotemT1','TotemT2','CastorF','CastorB','OQUA','MUON'),
        DetectorLevels= cms.vint32(3,3,3,3,3,3,4,3,3,3,3,3,3,3,3),
        EtaBoundaries = cms.vdouble(1.108,2.643,2.780,4.350,4.570,100.0),
        RegionTypes   = cms.vint32(0,1,0,1,0,1),
        Boundaries    = cms.vdouble(8050.,10860.,1595.,12800.,330.,16006.5),
        NBinEta      = cms.untracked.int32(250),
        NBinPhi      = cms.untracked.int32(180),
        MinEta       = cms.untracked.double(-7.0),
        MaxEta       = cms.untracked.double(-4.5)
    ),
    type = cms.string('MaterialBudgetForward')
))


