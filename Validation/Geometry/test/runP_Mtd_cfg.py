import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2_cff import Phase2

process = cms.Process("PROD",Phase2)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#Geometry
#
process.load("Configuration.Geometry.GeometryExtended2026D95Reco_cff")

#Magnetic Field
#
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Output of events, etc...
#
# Explicit note : since some histos/tree might be dumped directly,
#                 better NOT use PoolOutputModule !
# Detector simulation (Geant4-based)
#
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
#        threshold = cms.untracked.string('DEBUG'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet( ## but FwkJob category - those unlimitted
            limit = cms.untracked.int32(-1)
        ),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1000),
            limit = cms.untracked.int32(0)
        ),
        MaterialBudget = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
    ),
    categories = cms.untracked.vstring('FwkJob','FwkReport','MaterialBudget','G4cout','G4cerr'),
    debugModules = cms.untracked.vstring('g4SimHits'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:single_neutrino_random.root')
)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(-1)
    input = cms.untracked.int32(10000)
)

process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.StackingAction.TrackNeutrino = cms.bool(True)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('MaterialBudgetAction'),
    MaterialBudgetAction = cms.PSet(
        HistosFile = cms.string('matbdg_Mtd.root'),
        AllStepsToTree = cms.bool(False),
        HistogramList = cms.string('Mtd'),
        SelectedVolumes = cms.vstring('BarrelTimingLayer','EndcapTimingLayer'),
        # string TextFile = "None"          # "None" means this option 
        TreeFile = cms.string('None'),
        StopAfterProcess = cms.string('None'),
        TextFile = cms.string('None')
    )
))

process.g4SimHits.G4Commands = cms.vstring("/material/g4/printMaterial")
