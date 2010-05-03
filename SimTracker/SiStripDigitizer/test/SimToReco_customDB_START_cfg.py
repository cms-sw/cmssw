import FWCore.ParameterSet.Config as cms

process = cms.Process("MYRECO")

process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_37Y_V0::All'


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

   fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0015/D4146A66-0E4D-DF11-8E10-002618943984.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0015/BCF46584-D24C-DF11-8895-00261894398D.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0015/94C5841E-DA4C-DF11-9B1E-0018F3D0970A.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0015/44C3BEC1-D14C-DF11-B7D0-002618943852.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/B8CC20C4-D14C-DF11-9ED5-00248C0BE018.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/B0ADAAFE-CF4C-DF11-881A-001A92971B5E.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/AA131A2E-D14C-DF11-BC2B-002618943885.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/9CC9E4AC-D04C-DF11-B234-001A928116CE.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/7C819E0B-D04C-DF11-9C62-001A928116C8.root',
    '/store/relval/CMSSW_3_7_0_pre1/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/601377A3-D04C-DF11-8D07-002618943856.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

### new payloads:
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# noise payloads (for the clusterizer and the digitizer as well)
process.SiStripNoises = cms.ESSource("PoolDBESSource",process.CondDBSetup,
                             connect = cms.string("sqlite_file:dbfile_noise.db"),
                             toGet = cms.VPSet(cms.PSet(record = cms.string("SiStripNoisesRcd"),
                                                        tag = cms.string("SiStripNoiseNormalizedWithGain"))))
process.es_prefer_SiStripNoises = cms.ESPrefer("PoolDBESSource","SiStripNoises")

# gain payload for the clusterizer
process.SiStripGainReco = cms.ESSource("PoolDBESSource",process.CondDBSetup,
                           connect = cms.string('sqlite_file:dbfile_gainFromData.db'),
                           toGet = cms.VPSet(cms.PSet( record = cms.string('SiStripApvGainRcd'),
                                                       tag = cms.string('SiStripApvGain_gaussian'))))
process.es_prefer_SiStripGainReco = cms.ESPrefer("PoolDBESSource", "SiStripGainReco")

# gain payload for the digitizer
process.SiStripGainSim = cms.ESSource("PoolDBESSource",process.CondDBSetup,
                           connect = cms.string('sqlite_file:dbfile_gainFromData.db'),
                           toGet = cms.VPSet(cms.PSet( record = cms.string('SiStripApvGainSimRcd'),
                                                       tag = cms.string('SiStripApvGain_default'))))
process.es_prefer_SiStripGainSim = cms.ESPrefer("PoolDBESSource", "SiStripGainSim")

###


#Parameters for Simulation Digitizer
process.simSiStripDigis.ZeroSuppression = cms.bool(True)
process.simSiStripDigis.Noise = cms.bool(True)
process.simSiStripDigis.TrackerConfigurationFromDB = cms.bool(False) ## to use ConfDB


process.SiStripDigiToRaw.InputDigiLabel = cms.string('ZeroSuppressed')
process.SiStripDigiToRaw.FedReadoutMode = cms.string('ZERO_SUPPRESSED')

# Output definition

#process.output = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('drop *',
#    #'keep *_*_*_RECO'
#    'keep *_rawDataCollector_*_*',
#    'keep SiStripDigi*_*_*_MYRECO',
#    'drop SiStripDigi*_*_*_HLT'
#    ),
#    fileName = cms.untracked.string('/tmp/giamman/SimRawDigi_RelValMuPt10_OnlyTRK_customDB.root')
#)
process.output = cms.OutputModule("PoolOutputModule",
                                  splitLevel = cms.untracked.int32(0),
                                  outputCommands = process.RECODEBUGEventContent.outputCommands,
                                  fileName = cms.untracked.string('SimToReco_RelValMuPt10_customDB_START.root'),
                                  dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RECO'),
    filterName = cms.untracked.string('')
    )
                                  )


####

process.redigi_step = cms.Path(process.mix*process.simSiStripDigis)

process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction_withPixellessTk)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)



#process.p = cms.Path(process.mix*process.simSiStripDigis)
#process.outpath = cms.EndPath(process.output)





