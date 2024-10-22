import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process('PROD',Run3_DDD)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.Geometry.GeometryExtended2021Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
if hasattr(process,'MessageLogger'):
    process.MessageLogger.MaterialBudget=dict()

process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring('file:single_neutrino_random.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('matbdgHCAL_run3.root')
)

process.load("Validation.Geometry.materialBudgetHcalAnalysis_cfi")

process.simulation_step = cms.Path(process.g4SimHits)
process.analysis_step = cms.Path(process.materialBudgetHcalAnalysis)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetHcalProducer = cms.PSet(
        RMax         = cms.untracked.double(5.0),
        ZMax         = cms.untracked.double(14.0),
        Fromdd4hep   = cms.untracked.bool(False),
        EtaMinP      = cms.untracked.double(5.2),
        EtaMaxP      = cms.untracked.double(0.0),
        PrintSummary = cms.untracked.bool(True),
        Name         = cms.untracked.string('Hcal')
    ),
    type = cms.string('MaterialBudgetHcalProducer')
))

# Schedule definition
process.schedule = cms.Schedule(process.simulation_step,
                                process.analysis_step,
                                )
