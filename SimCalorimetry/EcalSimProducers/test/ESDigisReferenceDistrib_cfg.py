# The following comments couldn't be translated into the new config version:

# DQM services
# service = DaqMonitorROOTBackEnd{ }

import FWCore.ParameterSet.Config as cms

process = cms.Process("ESRefDistrib")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# geometry (Only Ecal)
process.load("Geometry.EcalCommonData.EcalOnly_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# run simulation, with EcalHits Validation specific watcher 
process.load("SimG4Core.Application.g4SimHits_cfi")

# Mixing Module
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

# Reconstruction geometry service
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

# use trivial ESProducer for tests
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

# ECAL 'slow' digitization sequence
process.load("SimCalorimetry.Configuration.ecalDigiSequenceComplete_cff")

# Producing the histos
process.load("SimCalorimetry.EcalSimProducers.ESDigisReferenceDistrib_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876),
        ecalUnsuppressedDigis = cms.untracked.uint32(12345)
    ),
    sourceSeed = cms.untracked.uint32(98765)
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

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
    Verbosity = cms.untracked.int32(0)
)

process.USER = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('./pedestal.root')
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.digis = cms.Sequence(process.ecalDigiSequenceComplete*process.randomEngineStateProducer*process.ESDigisReferenceDistrib)
process.p1 = cms.Path(process.g4SimHits*process.mix*process.digis)
process.outpath = cms.EndPath(process.USER)
process.g4SimHits.Generator.HepMCProductLabel = 'source'

