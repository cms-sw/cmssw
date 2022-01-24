import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process('PROD',Run3_DDD)
process.load('SimG4Core.PrintGeomInfo.cmsExtendedGeometry2021_cfi')
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff")
process.load("Geometry.EcalCommonData.ecalSimulationParameters_cff")
process.load("Geometry.HcalCommonData.hcalDDDSimConstants_cff")
process.load("Geometry.HcalCommonData.hcalDDDRecConstants_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.MuonNumbering.muonOffsetESProducer_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("Validation.Geometry.materialBudgetVolume_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
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
    fileName = cms.string('matbdgRun3dddBigXML.root')
)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False

process.load("Validation.Geometry.materialBudgetVolumeAnalysis_cfi")
process.p1 = cms.Path(process.g4SimHits+process.materialBudgetVolumeAnalysis)
