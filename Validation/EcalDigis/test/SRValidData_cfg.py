import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSelectiveReadoutValid")

#Number of events to process
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#Input files
process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/c/ccecal/BEAM/Skims/Collisions7TeV/run132440/express/bit40or41skim_startingFromSkim2_run132440.root')
     fileNames = cms.untracked.vstring('file:/tmp/pgras/bit40or41skim_startingFromSkim2_run132440.root')
)

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

#process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

###process.GlobalTag.globaltag = 'MC_3XY_V18::All'
process.GlobalTag.globaltag = 'GR10_P_V4::All'
 
# ECAL digitization sequence
process.load("SimCalorimetry.Configuration.ecalDigiSequence_cff")
process.simEcalDigis.trigPrimProducer = cms.string('ecalEBunpacker')
process.simEcalDigis.trigPrimCollection =  cms.string('EcalTriggerPrimitives')
process.simEcalDigis.digiProducer = cms.string('ecalEBunpacker')
process.simEcalDigis.EBdigiCollection = cms.string('ebDigis')
process.simEcalDigis.EEdigiCollection = cms.string('eeDigis')

# Defines Ecal seletive readout validation module, ecalSelectiveReadoutValidation:
process.load("Validation.EcalDigis.ecalSelectiveReadoutValidation_cfi")
process.ecalSelectiveReadoutValidation.outputFile = 'srvalid_hists.root'
#process.ecalSelectiveReadoutValidation.EbDigiCollection = cms.InputTag("selectDigi", "selectedEcalEBDigiCollection");
#process.ecalSelectiveReadoutValidation.EeDigiCollection = cms.InputTag("selectDigi", "selectedEcalEEDigiCollection");
#process.ecalSelectiveReadoutValidation.EbSrFlagCollection = cms.InputTag("ecalDigis","");
#process.ecalSelectiveReadoutValidation.EeSrFlagCollection = cms.InputTag("ecalDigis","");

process.ecalSelectiveReadoutValidation.EbDigiCollection = cms.InputTag("ecalEBunpacker", "ebDigis")
process.ecalSelectiveReadoutValidation.EeDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis")
#process.ecalSelectiveReadoutValidation.EbUnsuppressedDigiCollection = cms.InputTag("simEcalUnsuppressedDigis")
#process.ecalSelectiveReadoutValidation.EeUnsuppressedDigiCollection = cms.InputTag("simEcalUnsuppressedDigis")
process.ecalSelectiveReadoutValidation.EbSrFlagCollection = cms.InputTag("ecalEBunpacker","")
process.ecalSelectiveReadoutValidation.EeSrFlagCollection = cms.InputTag("ecalEBunpacker","")
process.ecalSelectiveReadoutValidation.EbSrFlagFromTTCollection = cms.InputTag("simEcalDigis","ebSrFlags")
process.ecalSelectiveReadoutValidation.EeSrFlagFromTTCollection = cms.InputTag("simEcalDigis","eeSrFlags")
#rocess.ecalSelectiveReadoutValidation.EbSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEB")
#process.ecalSelectiveReadoutValidation.EeSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEE")
process.ecalSelectiveReadoutValidation.TrigPrimCollection = cms.InputTag("ecalEBunpacker", "EcalTriggerPrimitives")
process.ecalSelectiveReadoutValidation.EbRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
#process.ecalSelectiveReadoutValidation.EeRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
#process.ecalSelectiveReadoutValidation.FEDRawCollection = cms.InputTag("source")
#process.ecalSelectiveReadoutValidation.EventHeaderCollection = cms.InputTag("ecalEBunpacker")
process.ecalSelectiveReadoutValidation.ecalDccZs1stSample = 3
process.ecalSelectiveReadoutValidation.dccWeights = [ -1.1865, 0.0195, 0.2900, 0.3477, 0.3008, 0.2266 ]
process.ecalSelectiveReadoutValidation.histDir = ''
process.ecalSelectiveReadoutValidation.histograms = [ 'all' ]
#[ "h2ChOcc",
#"h2FRORu",
#"h2ForcedRu",
#"h2ForcedTtf",
#"h2HiTtf",
#"h2LtTtf",
#"h2MiTtf",
#"h2Tp",
#"h2TpVsEtSum",
#"h2TtfVsEtSum",
#"h2TtfVsTp",
#"h2Zs1Ru",
#"hCompleteZSMap",
#"hCompleteZSRate",
#"hCompleteZsCnt",
#"hDccHiVol",
#"hDccLiVol",
#"hDccVol",
#"hDccVolFromData",
#"hDroppedFROCnt",
#"hDroppedFROMap",
#"hDroppedFRORateMap",
#"hEbEMean",
#"hEbFROCnt",
#"hEbRecE",
#"hEbZsErrCnt",
#"hEbZsErrType1Cnt",
#"hEeEMean",
#"hEeFROCnt",
#"hEeRecE",
#"hEeZsErrCnt",
#"hEeZsErrType1Cnt",
#"hFROCnt",
#"hIncompleteFROCnt",
#"hIncompleteFROMap",
#"hIncompleteFRORateMap",
#"hSRAlgoErrorMap",
#"hTp",
#"hTtf",
#"hVol",
#"hVolB",
#"hVolBHI",
#"hVolBLI",
#"hVolE",
#"hVolEHI",
#"hVolELI",
#"hVolHI",
#"hVolLI",
#"hZsErrCnt",
#"hZsErrType1Cnt",
#"zsEbHiFIRemu",
#"zsEbLiFIRemu",
#"zsEeHiFIRemu",
#"zsEeLiFIRemu"
#]
#process.ecalSelectiveReadoutValidation.useEventRate = False
process.ecalSelectiveReadoutValidation.LocalReco = cms.bool(False)

# ECAL Unpacker:
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.ecalEBunpacker.silentMode = cms.untracked.bool(True)
    

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

#process.load("pgras.ListCollection.ListCollection_cfi")

process.p1 = cms.Path(process.ecalEBunpacker*process.simEcalDigis*process.ecalSelectiveReadoutValidation)
process.DQM.collectorHost = ''
