import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('PROD',Run3)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.Geometry.GeometryExtended2021_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.load('FWCore.MessageService.MessageLogger_cfi')
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

process.p1 = cms.Path(process.g4SimHits)
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetHcal = cms.PSet(
        FillHisto    = cms.untracked.bool(True),
        PrintSummary = cms.untracked.bool(True),
        DoHCAL       = cms.untracked.bool(True),
        NBinPhi      = cms.untracked.int32(180),
        NBinEta      = cms.untracked.int32(260),
        MaxEta       = cms.untracked.double(5.2),
        EtaLow       = cms.untracked.double(-5.2),
        EtaHigh      = cms.untracked.double(5.2),
        EtaMinP      = cms.untracked.double(5.2),
        EtaMaxP      = cms.untracked.double(0.0),
#       EtaMinP      = cms.untracked.double(1.39),
#       EtaMaxP      = cms.untracked.double(1.42),
#       EtaMinP      = cms.untracked.double(2.90),
#       EtaMaxP      = cms.untracked.double(3.00),
        EtaLowMin    = cms.untracked.double(0.783),
        EtaLowMax    = cms.untracked.double(0.870),
        EtaMidMin    = cms.untracked.double(2.650),
        EtaMidMax    = cms.untracked.double(2.868),
        EtaHighMin   = cms.untracked.double(2.868),
        EtaHighMax   = cms.untracked.double(3.000),
        RMax         = cms.untracked.double(5.0),
        ZMax         = cms.untracked.double(14.0),
        Fromdd4hep   = cms.untracked.bool(False)
    ),
    type = cms.string('MaterialBudgetHcal')
))


