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

process.poolDBESSource = cms.ESSource("PoolDBESSource",
 BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
 DBParameters = cms.PSet(
      messageLevel = cms.untracked.int32(2),
      authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
  ),
  timetype = cms.untracked.string('runnumber'),
  connect = cms.string('frontier://FrontierProd/CMS_COND_31X_STRIP'),
  toGet = cms.VPSet(cms.PSet(
      record = cms.string('SiStripNoisesRcd'),
      tag = cms.string('SiStripNoise_CRAFT09_DecMode_ForTrackerSim')
  ),
   cms.PSet(
      record = cms.string('SiStripPedestalsRcd'),
      tag = cms.string('SiStripPedestal_CRAFT09_DecMode_ForTrackerSim')
  ),
   cms.PSet(
      record = cms.string('SiStripFedCablingRcd'),
      tag = cms.string('SiStripFedCabling_CRAFT09_ForTrackerSim')
  ),
  cms.PSet(
      record = cms.string('SiStripBadChannelRcd'),
      tag = cms.string('SiStripBadChannelsFromO2O_CRAFT09_DecMode_ForTrackerSim')
  )
)
) 


process.es_prefer_my =cms.ESPrefer("PoolDBESSource","poolDBESSource")


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
    'keep *_rawDataCollector_*_*',
    'keep SiStripDigi*_*_*_*'
    ),
    fileName = cms.untracked.string('SimRawDigi_RelValMuPt10_OnlyTRK_TRKinVR.root')
)


#Parameters for Simulation Digitizer
#Zero Supp FALSE = VR
process.simSiStripDigis.ZeroSuppression = cms.bool(False)
process.simSiStripDigis.Noise = cms.bool(True)
process.simSiStripDigis.TrackerConfigurationFromDB = cms.bool(True)

#VR generation blocks to be activated
process.simSiStripDigis.SingleStripNoise = cms.bool(True) #if Noise = FALSE, no noise is applied
process.simSiStripDigis.RealPedestals = cms.bool(True)
process.simSiStripDigis.CommonModeNoise = cms.bool(True)
process.simSiStripDigis.APVSaturationFromHIP = cms.bool(True)
process.simSiStripDigis.BaselineShift = cms.bool(True)

#CMN RMSs
process.simSiStripDigis.cmnRMStib = cms.double(5.92)
process.simSiStripDigis.cmnRMStob = cms.double(1.08)
process.simSiStripDigis.cmnRMStid = cms.double(3.08)
process.simSiStripDigis.cmnRMStec = cms.double(2.44)
	   					   
							   
process.SiStripDigiToRaw.InputDigiLabel = cms.string('VirginRaw')
process.SiStripDigiToRaw.FedReadoutMode = cms.string('VIRGIN_RAW')

process.p = cms.Path(process.mix*process.simSiStripDigis) #Generate RAWDigiOnly tracker 

process.outpath = cms.EndPath(process.output)





