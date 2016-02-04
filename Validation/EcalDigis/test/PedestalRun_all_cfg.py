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
process.load("Validation.EcalDigis.ecalUnsuppressedDigisValidationSequence_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        PartID = cms.untracked.vint32(14),
        MaxEta = cms.untracked.double(3.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.0),
        MinE = cms.untracked.double(9.99),
        MinPhi = cms.untracked.double(-3.14159265359), ## in radians

        MaxE = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0) ## set to 1 (or greater)  for printouts

)

process.USER = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('PedestalRun_all_EcalValidation.root')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        simEcalUnsuppressedDigis = cms.untracked.uint32(12345)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.simhits = cms.Sequence(process.g4SimHits*process.ecalSimHitsValidationSequence)
process.digis = cms.Sequence(process.mix*process.simEcalUnsuppressedDigis*process.ecalUnsuppressedDigisValidationSequence)
process.p1 = cms.Path(process.simhits)
process.p2 = cms.Path(process.digis)
process.p4 = cms.Path(process.randomEngineStateProducer)
process.outpath = cms.EndPath(process.USER)
process.schedule = cms.Schedule(process.p1,process.p2,process.p4,process.outpath)

process.DQM.collectorHost = ''
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    instanceLabel = cms.untracked.string('EcalValidInfo'),
    type = cms.string('EcalSimHitsValidProducer'),
    verbose = cms.untracked.bool(False)
))

