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

# ECAL unsuppressed digis
process.load("SimCalorimetry.EcalSimProducers.ecaldigi_cfi")

# ECAL unsuppressed digis validation sequence
process.load("Validation.EcalDigis.ecalDigisValidation_cfi")

process.load("Validation.EcalDigis.ecalBarrelDigisValidation_cfi")

process.load("Validation.EcalDigis.ecalEndcapDigisValidation_cfi")

process.load("Validation.EcalDigis.ecalPreshowerDigisValidation_cfi")

# ECAL (unsuppressed) LocalReco sequence 
process.load("RecoLocalCalo.EcalRecProducers.ecalLocalRecoSequence_cff")

# ECAL (unsuppressed) rechits validation sequence
process.load("Validation.EcalRecHits.ecalUnsuppressedRecHitsValidationSequence_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:Photon_E30GeV_all.root')
)

process.USER = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *', 
        'drop *_simEcalUnsuppressedDigis_*_*', 
        'drop *_simEcalDigis_*_*', 
        'drop *_simEcalPreshowerDigis_*_*', 
        'drop *_ecalRecHit_*_*', 
        'drop *_ecalPreshowerRecHit_*_*'),
    fileName = cms.untracked.string('Photon_E30GeV_all_EcalUnsuppressedValidation.root')
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
process.digis = cms.Sequence(process.mix*process.simEcalUnsuppressedDigis)
process.digisVal = cms.Sequence(process.ecalDigisValidation*process.ecalBarrelDigisValidation*process.ecalEndcapDigisValidation*process.ecalPreshowerDigisValidation)
process.rechits = cms.Sequence(process.ecalLocalRecoSequence*process.ecalUnsuppressedRecHitsValidationSequence)
process.p1 = cms.Path(process.simhits)
process.p2 = cms.Path(process.digis)
process.p3 = cms.Path(process.digisVal)
process.p4 = cms.Path(process.rechits)
process.p5 = cms.Path(process.randomEngineStateProducer)
process.outpath = cms.EndPath(process.USER)
process.schedule = cms.Schedule(process.p1,process.p2,process.p3,process.p4,process.p5,process.outpath)

process.DQM.collectorHost = ''
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    instanceLabel = cms.untracked.string('EcalValidInfo'),
    type = cms.string('EcalSimHitsValidProducer'),
    verbose = cms.untracked.bool(False)
))
process.simEcalUnsuppressedDigis.EBdigiCollection = 'ebUnsuppDigis'
process.simEcalUnsuppressedDigis.EEdigiCollection = 'eeUnsuppDigis'
process.simEcalUnsuppressedDigis.ESdigiCollection = 'esUnsuppDigis'
process.ecalDigisValidation.EBdigiCollection = 'simEcalUnsuppressedDigis:ebUnsuppDigis'
process.ecalDigisValidation.EEdigiCollection = 'simEcalUnsuppressedDigis:eeUnsuppDigis'
process.ecalDigisValidation.ESdigiCollection = 'simEcalUnsuppressedDigis:esUnsuppDigis'
process.ecalBarrelDigisValidation.EBdigiCollection = 'simEcalUnsuppressedDigis:ebUnsuppDigis'
process.ecalEndcapDigisValidation.EEdigiCollection = 'simEcalUnsuppressedDigis:eeUnsuppDigis'
process.ecalPreshowerDigisValidation.ESdigiCollection = 'simEcalUnsuppressedDigis:esUnsuppDigis'
process.ecalWeightUncalibRecHit.EBdigiCollection = 'simEcalUnsuppressedDigis:ebUnsuppDigis'
process.ecalWeightUncalibRecHit.EEdigiCollection = 'simEcalUnsuppressedDigis:eeUnsuppDigis'
process.ecalPreshowerRecHit.ESdigiCollection = 'simEcalUnsuppressedDigis:esUnsuppDigis'

