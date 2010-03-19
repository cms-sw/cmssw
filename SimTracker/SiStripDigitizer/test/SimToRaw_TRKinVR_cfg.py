import FWCore.ParameterSet.Config as cms

process = cms.Process("MYRECO")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.L1Emulator_cff")
#process.load('Configuration/StandardSequences/SimL1Emulator_cff')
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'GR09_P_V5::All'
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

#process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
#  ReduceGranularity = cms.bool(False),
#  PrintDebugOutput = cms.bool(False),
#  UseEmptyRunInfo = cms.bool(False),
#  ListOfRecordToMerge = cms.VPSet(cms.PSet(
#  record = cms.string('SiStripBadChannelRcd'),
#  tag = cms.string('')
#  ),
#  cms.PSet(
#  record = cms.string('SiStripDetCablingRcd'),
#  tag = cms.string('')
#  )
#)
#)

#process.es_prefer_myquality=cms.ESPrefer("SiStripQualityESProducer","SiStripQualityESProducer")

#process.sistripconn = cms.ESProducer("SiStripConnectivity")

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
#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('*'),
#    destinations = cms.untracked.vstring('')
#    fwkJobReports = cms.untracked.vstring('FrameworkJobReport.xml')
#)


process.source = cms.Source("PoolSource",
   #fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0007/FAE9ED85-9078-DE11-8F39-001D09F231C9.root'),
   #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/y/yilmaz/share/Hydjet_MinBias_4TeV_312.root'), #this is the original location 

   fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_5_4/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0004/46A38EF8-2C2C-DF11-9E99-00261894388D.root')
#ths is for B0 events      
#fileNames = cms.untracked.vstring(#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/1282CFD8-7284-DE11-A438-000423D174FE.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/6C90B08C-7084-DE11-B353-000423D9853C.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/76861267-7084-DE11-9487-000423D987FC.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/7AFDBB0D-6984-DE11-8D31-000423D6CAF2.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/84991F14-7C84-DE11-9A31-000423D99AAA.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/A066DF06-6884-DE11-80CE-000423D98750.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/A63C0F42-7184-DE11-AF76-001617DC1F70.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/C2AC8D30-6984-DE11-87F5-000423D6B42C.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/E6D206A5-6884-DE11-A72A-000423D95030.root',
#'/store/relval/CMSSW_3_2_4/RelValHydjetQ_B0_4TeV/GEN-SIM-RAW/MC_31X_V3-v1/0010/FC70A0E1-6A84-DE11-AE66-000423D98DB4.root')
#

#ths is for MinBias events      
#fileNames = cms.untracked.vstring('file:16F98AA0-9E13-DF11-982F-0030486733B4.root'
#'/store/relval/CMSSW_3_4_2/RelValHydjetQ_B0_4TeV/GEN-SIM-RECO/MC_3XY_V15-v2/0011/16F98AA0-9E13-DF11-982F-0030486733B4.root',
#'/store/relval/CMSSW_3_4_2/RelValHydjetQ_B0_4TeV/GEN-SIM-RECO/MC_3XY_V15-v2/0011/18291594-9E13-DF11-B9B0-0030487A3DE0.root'
#)
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
    fileName = cms.untracked.string('SimRawDigi_RelValMuPt10_OnlyTRK_TRKinVR_onlyHIP.root')
)


#Parameters for Simulation Digitizer
#Zero Supp FALSE = VR
process.simSiStripDigis.ZeroSuppression = cms.bool(False)
process.simSiStripDigis.Noise = cms.bool(True)
process.simSiStripDigis.TrackerConfigurationFromDB = cms.bool(True)

#VR generation blocks to be activated
process.simSiStripDigis.SingleStripNoise = cms.bool(False) #if Noise = FALSE, no noise is applied
process.simSiStripDigis.RealPedestals = cms.bool(True)
process.simSiStripDigis.CommonModeNoise = cms.bool(False)
process.simSiStripDigis.APVSaturationFromHIP = cms.bool(False)
process.simSiStripDigis.BaselineShift = cms.bool(False)

#CMN RMSs
process.simSiStripDigis.cmnRMStib = cms.double(5.92)
process.simSiStripDigis.cmnRMStob = cms.double(1.08)
process.simSiStripDigis.cmnRMStid = cms.double(3.08)
process.simSiStripDigis.cmnRMStec = cms.double(2.44)
	   					   
							   
process.SiStripDigiToRaw.InputDigiLabel = cms.string('VirginRaw')
process.SiStripDigiToRaw.FedReadoutMode = cms.string('VIRGIN_RAW')

#process.SiStripDigiToRaw.InputDigiLabel = cms.string('ZeroSuppressed')
#process.SiStripDigiToRaw.FedReadoutMode = cms.string('ZERO_SUPPRESSED')


####Remember to Set the right Conditions and the right path
#process.p = cms.Path(process.mix*process.simSiStripDigis*process.SiStripDigiToRaw*process.rawDataCollector) #Generate RAW-Only tracker
#process.p = cms.Path(process.mix*process.doAllDigi*process.L1Emulator*process.DigiToRaw)                   #Generate RAW-ALL CMS
#process.p = cms.Path(process.mix*process.doAllDigi*process.L1Emulator)                                     #Generate RAWDigis ALL CMS 
process.p = cms.Path(process.mix*process.simSiStripDigis) #Generate RAWDigiOnly tracker 

process.outpath = cms.EndPath(process.output)





