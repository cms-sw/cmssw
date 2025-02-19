import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalHitsValid")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# geometry (Only Ecal)
process.load("Geometry.EcalCommonData.EcalOnly_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# run simulation, with EcalHits Validation specific watcher 
process.load("SimG4Core.Application.g4SimHits_cfi")

# ECAL hits validation sequence
process.load("Validation.EcalHits.ecalSimHitsValidationSequence_cff")

# End of process
process.load("Configuration.StandardSequences.EndOfProcess_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:Photon_E30GeV_all.root')
)

process.USER = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('Photon_E30GeV_all_EcalValidation.root')
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(9876)
    )
)

process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.simhits = cms.Sequence(process.g4SimHits*process.ecalSimHitsValidationSequence)
process.p1 = cms.Path(process.simhits)
process.p4 = cms.Path(process.randomEngineStateProducer)
process.p5 = cms.Path(process.endOfProcess)
process.outpath = cms.EndPath(process.USER)
process.schedule = cms.Schedule(process.p1,process.p4,process.p5,process.outpath)

process.DQM.collectorHost = ''
process.g4SimHits.Generator.HepMCProductLabel = 'source'
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    instanceLabel = cms.untracked.string('EcalValidInfo'),
    type = cms.string('EcalSimHitsValidProducer'),
    verbose = cms.untracked.bool(False)
))



