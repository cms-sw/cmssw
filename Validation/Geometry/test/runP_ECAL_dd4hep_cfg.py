import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
process = cms.Process('PROD',Run3_dd4hep)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#Geometry
#
process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')

#Magnetic Field
#
process.load("Configuration.StandardSequences.MagneticField_cff")

# Output of events, etc...
#
# Explicit note : since some histos/tree might be dumped directly,
#                 better NOT use PoolOutputModule !
# Detector simulation (Geant4-based)
#
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.load('FWCore.MessageService.MessageLogger_cfi')
#if hasattr(process,'MessageLogger'):
#    process.MessageLogger.MaterialBudget=dict()

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:single_neutrino_random.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('MaterialBudgetAction'),
    MaterialBudgetAction = cms.PSet(
        HistosFile = cms.string('matbdg_ECAL_DD4hep.root'),
        AllStepsToTree = cms.bool(False),
        HistogramList = cms.string('ECAL'),
        SelectedVolumes = cms.vstring('ECAL'),
        # string TextFile = "None"          # "None" means this option 
        TreeFile = cms.string('None'),
        StopAfterProcess = cms.string('None'),
        TextFile = cms.string('matbdg_ECAL_DD4hep.txt')
    )
))


