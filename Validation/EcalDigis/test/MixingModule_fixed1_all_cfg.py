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

# Mixing Module
process.load("SimGeneral.MixingModule.mixLowLumPU_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

# ECAL digitization sequence
process.load("SimCalorimetry.Configuration.ecalDigiSequence_cff")

# ECAL digis validation sequence
process.load("Validation.EcalDigis.ecalDigisValidationSequence_cff")

# ECAL Mixing Module specific validation
process.load("Validation.EcalDigis.ecalMixingModuleValidation_cfi")

# End of process
process.load("Configuration.StandardSequences.EndOfProcess_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        ecalMixingModuleValidation = cms.untracked.uint32(12345),
        simEcalUnsuppressedDigis = cms.untracked.uint32(12345),
        mix = cms.untracked.uint32(1234)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.source = cms.Source("MCFileSource",
    fileNames = cms.untracked.vstring('file:Photon_E30GeV_fixed.dat')
)

process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('MixingModule_fixed1_all_EcalValidation.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.simhits = cms.Sequence(process.g4SimHits)
process.digis = cms.Sequence(process.mix*process.ecalDigiSequence*process.ecalDigisValidationSequence*process.ecalMixingModuleValidation)
process.p1 = cms.Path(process.simhits)
process.p2 = cms.Path(process.digis)
process.p4 = cms.Path(process.randomEngineStateProducer)
process.p5 = cms.Path(process.endOfProcess)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p1,process.p2,process.p4,process.p5,process.outpath)

process.DQM.collectorHost = ''
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.mix.input.fileNames = ['file:MixingModule_noPileup_all_EcalValidation.root']
process.mix.input.type = 'fixed'
process.mix.minBunch = -5
process.mix.maxBunch = 3
process.mix.input.nbPileupEvents = cms.PSet(
    averageNumber = cms.double(1.0)
)

