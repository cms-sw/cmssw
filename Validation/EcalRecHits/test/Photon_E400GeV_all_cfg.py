import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalHitsValid")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# geometry (Only Ecal)
process.load("Geometry.EcalCommonData.EcalOnly_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# run simulation, with EcalHits Validation specific watcher 
process.load("SimG4Core.Application.g4SimHits_cfi")

# ECAL hits validation sequence
process.load("Validation.EcalHits.ecalSimHitsValidationSequence_cff")

# Mixing Module
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

# ECAL digitization sequence
process.load("SimCalorimetry.Configuration.ecalDigiSequence_cff")

# ECAL digis validation sequence
process.load("Validation.EcalDigis.ecalDigisValidationSequence_cff")

# ECAL LocalReco sequence 
process.load("RecoLocalCalo.EcalRecProducers.ecalLocalRecoSequence_cff")

# ECAL rechits validation sequence
process.load("Validation.EcalRecHits.ecalRecHitsValidationSequence_cff")

# End of process
process.load("Configuration.StandardSequences.EndOfProcess_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:Photon_E400GeV_all.root')
)

process.USER = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop *_simEcalUnsuppressedDigis_*_*', 
        'drop *_simEcalDigis_*_*', 
        'drop *_simEcalPreshowerDigis_*_*', 
        'drop *_ecalRecHit_*_*', 
        'drop *_ecalPreshowerRecHit_*_*'),
    fileName = cms.untracked.string('Photon_E400GeV_all_EcalValidation.root')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        simEcalUnsuppressedDigis = cms.untracked.uint32(12345)
    )
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.simhits = cms.Sequence(process.g4SimHits*process.ecalSimHitsValidationSequence)
process.digis = cms.Sequence(process.mix*process.ecalDigiSequence*process.ecalDigisValidationSequence)
process.rechits = cms.Sequence(process.ecalLocalRecoSequence*process.ecalRecHitsValidationSequence)
process.p1 = cms.Path(process.simhits)
process.p2 = cms.Path(process.digis)
process.p3 = cms.Path(process.rechits)
process.p4 = cms.Path(process.randomEngineStateProducer)
process.p5 = cms.Path(process.endOfProcess)
process.outpath = cms.EndPath(process.USER)
process.schedule = cms.Schedule(process.p1,process.p2,process.p3,process.p4,process.p5,process.outpath)

process.DQM.collectorHost = ''
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    instanceLabel = cms.untracked.string('EcalValidInfo'),
    type = cms.string('EcalSimHitsValidProducer'),
    verbose = cms.untracked.bool(False)
))
process.ecalUncalibRecHit.EBdigiCollection = 'simEcalDigis:ebDigis'
process.ecalUncalibRecHit.EEdigiCollection = 'simEcalDigis:eeDigis'
process.ecalPreshowerRecHit.ESdigiCollection = 'simEcalPreshowerDigis'

