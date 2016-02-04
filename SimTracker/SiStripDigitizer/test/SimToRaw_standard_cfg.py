import FWCore.ParameterSet.Config as cms

process = cms.Process("MYRECO")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_3XY_V24::All'


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
   moduleSeeds = cms.PSet(
        simMuonRPCDigis = cms.untracked.uint32(6),
        simEcalUnsuppressedDigis = cms.untracked.uint32(8),
        simSiStripDigis = cms.untracked.uint32(7),
        simSiStripDigisOrig = cms.untracked.uint32(7),
        mix = cms.untracked.uint32(4),
        simHcalUnsuppressedDigis = cms.untracked.uint32(9),
        simMuonCSCDigis = cms.untracked.uint32(6),
        VtxSmeared = cms.untracked.uint32(2),
        g4SimHits = cms.untracked.uint32(3),
        simMuonDTDigis = cms.untracked.uint32(6),
        simSiPixelDigis = cms.untracked.uint32(7)
    ),
    sourceSeed = cms.untracked.uint32(1) 
)


process.load("FWCore.MessageLogger.MessageLogger_cfi") 


process.source = cms.Source("PoolSource",

   fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_5_4/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0004/46A38EF8-2C2C-DF11-9E99-00261894388D.root')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *',
    #'keep *_*_*_RECO'
    'keep *_rawDataCollector_*_*',
    'keep SiStripDigi*_*_*_*'
    ),
    fileName = cms.untracked.string('SimRawDigi_RelValMuPt10_OnlyTRK_standard.root')
)


#Parameters for Simulation Digitizer
process.simSiStripDigis.ZeroSuppression = cms.bool(True)
process.simSiStripDigis.Noise = cms.bool(True)
process.simSiStripDigis.TrackerConfigurationFromDB = cms.bool(False)


process.SiStripDigiToRaw.InputDigiLabel = cms.string('ZeroSuppressed')
process.SiStripDigiToRaw.FedReadoutMode = cms.string('ZERO_SUPPRESSED')


process.p = cms.Path(process.mix*process.simSiStripDigis)

process.outpath = cms.EndPath(process.output)





