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

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:single_neutrino_random.root')
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        MaterialBudget = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('*')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('matbdg_Calo.root')
)

process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudget = cms.PSet(
        DetectorTypes = cms.vstring('BeamPipe','Tracker','EM Calorimeter','Hadron Calorimeter','Muon System','Forward Shield'),
        Constituents  = cms.vint32(4,2,1,2,1,1),
        StackOrder    = cms.vint32(1,2,3,4,5,6),
        DetectorNames = cms.vstring('BEAM','BEAM1','BEAM2','BEAM3','Tracker','PLT','ECAL','CALO','VCAL','MUON','OQUA'),
        DetectorLevels= cms.vint32(3,3,3,3,3,3,4,3,3,3,3),
        EtaBoundaries = cms.vdouble(1.108,2.643,2.780,4.350,4.570,100.0),
        RegionTypes   = cms.vint32(0,1,0,1,0,1),
        Boundaries    = cms.vdouble(8050.,10860.,1595.,12800.,330.,16006.5),
        NBinEta      = cms.untracked.int32(250),
        NBinPhi      = cms.untracked.int32(180),
        MinEta       = cms.untracked.double(-5.0),
        MaxEta       = cms.untracked.double(5.0)
    ),
    type = cms.string('MaterialBudget')
))
