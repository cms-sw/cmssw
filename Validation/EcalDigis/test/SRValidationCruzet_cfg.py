import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSelectiveReadoutValid")

# Data input:
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/CRUZET2/Cosmics/RAW/v1/000/046/629/9A33ABCA-F037-DD11-8D85-000423D6BA18.root', 
        '/store/data/CRUZET2/Cosmics/RAW/v1/000/046/629/9CE2CDB9-F137-DD11-9840-000423D992DC.root', 
        '/store/data/CRUZET2/Cosmics/RAW/v1/000/046/629/F46E124B-F137-DD11-9FF4-000423D6B358.root')
)

# Number of events to process
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# ECAL data unpacker (raw to digi conversion):
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

# Initializes  MessageLogger:
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10
process.MessageLogger.suppressWarning = ['ecalSelectiveReadoutValidation']

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# Geometry (Only Ecal)
process.load("Geometry.EcalCommonData.EcalOnly_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

# DQM services
process.load("DQMServices.Core.DQM_cfg")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

# Defines Ecal seletive readout validation module, ecalSelectiveReadoutValidation:
process.load("Validation.EcalDigis.ecalSelectiveReadoutValidation_cfi")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(10)
)

process.p1 = cms.Path(process.ecalEBunpacker*process.ecalSelectiveReadoutValidation)

process.DQM.collectorHost = ''
process.ecalSelectiveReadoutValidation.EbDigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalSelectiveReadoutValidation.EeDigiCollection = 'ecalEBunpacker:eeDigis'
process.ecalSelectiveReadoutValidation.EbSrFlagCollection = 'ecalEBunpacker:'
process.ecalSelectiveReadoutValidation.EeSrFlagCollection = 'ecalEBunpacker:'
process.ecalSelectiveReadoutValidation.TrigPrimCollection = 'ecalEBunpacker:EcalTriggerPrimitives'
process.ecalSelectiveReadoutValidation.tpInGeV = False
process.ecalSelectiveReadoutValidation.outputFile = 'srvalid_hists.root'
process.ecalSelectiveReadoutValidation.histDir = ''
process.ecalSelectiveReadoutValidation.histograms = [
    "EBEventSize", #Barrel data volume;Event size (kB);Nevts
    "EBHighInterestPayload", #Barrel high interest crystal data payload;Event size (kB);Nevts
    "EBLowInterestPayload", #ECAL Barrel low interest crystal data payload;Event size (kB);Nevts
    "EEEventSize", #Endcap data volume;Event size (kB);Nevts
    "EEHighInterestPayload", #Endcap high interest crystal data payload;Event size (kB);Nevts
    "EELowInterestPayload", #Endcap low interest crystal data payload;Event size (kB);Nevts
    "EbZeroSupp1SRFlagMap", #Trigger tower read-out with ZS threshold 1;iPhi;iEta;Event count
    "EcalChannelOccupancy", #ECAL crystal channel occupancy after zero suppression;iX0 / iEta0+120 / iX0 + 310;iY0 / iPhi0 (starting from 0);Event count
    "EcalDccEventSize", #ECAL DCC event fragment size;Dcc id; <Event size> (kB)
    "EcalDccEventSizeComputed", #ECAL DCC event fragment size;Dcc id; <Event size> (kB)
    "EcalEventSize", #ECAL data volume;Event size (kB);Nevts
    "EcalFullReadoutSRFlagMap", #Full Read-out trigger tower;iPhi;iEta;Event count
    "EcalHighInterestPayload", #ECAL high interest crystal data payload;Event size (kB);Nevts
    "EcalHighInterestTriggerTowerFlagMap", #High interest trigger tower flags;iPhi;iEta;Event count
    "EcalLowInterestPayload", #ECAL low interest crystal data payload;Event size (kB);Nevts
    "EcalLowInterestTriggerTowerFlagMap", #Low interest trigger tower flags;iPhi;iEta;Event count
    "EcalMidInterestTriggerTowerFlagMap", #Mid interest trigger tower flags;iPhi;iEta;Event count
    "EcalReadoutUnitForcedBitMap", #ECAL readout unit with forced bit of SR flag on;iPhi;iEta;Event count
    "EcalTriggerPrimitiveEt", #Trigger primitive TT E_{T};E_{T} GeV;Event Count
    "EcalTriggerPrimitiveEtMap", #Trigger primitive;iPhi;iEta;Event count
    "EcalTriggerTowerFlag", #Trigger primitive TT flag;Flag number;Event count
    "hEbEMean", #EE <E_hit>;event #;<E_hit> (GeV)
    "hEbNoZsRecVsSimE", #Crystal no-zs simulated vs reconstructed energy;Esim (GeV);Erec GeV);Event count
    "hEbRecE", #Crystal reconstructed energy;E (GeV);Event count
    "hEbRecEHitXtal", #EB rec energy of hit crystals
    "hEeEMean", #EE <E_hit>;event #;<E_hit> (GeV)
    "hEeRecE", #EE crystal reconstructed energy;E (GeV);Event count
    "hEeRecEHitXtal", #EE rec energy of hit crystals
    "hEeRecVsSimE", #EE crystal simulated vs reconstructed energy;Esim (GeV);Erec GeV);Event count
    "tpVsEtSum", #Trigger primitive Et (TP) vs #sumE_{T};E_{T} (sum) (GeV);E_{T} (TP) (GeV)
    "ttfVsEtSum", #Trigger tower flag vs #sumE_{T};E_{T}(TT) (GeV);TTF
    "ttfVsTp", #Trigger tower flag vs TP;E_{T}(TT) (GeV);Flag number
    "zsEbHiFIRemu", #Emulated ouput of ZS FIR filter for EB high interest crystals;ADC count*4;Event count
    "zsEbLiFIRemu", #Emulated ouput of ZS FIR filter for EB low interest crystals;ADC count*4;Event count
    "zsEeHiFIRemu", #Emulated ouput of ZS FIR filter for EE high interest crystals;ADC count*4;Event count
    "zsEeLiFIRemu", #Emulated ouput of ZS FIR filter for EE low interest crystals;ADC count*4;Event count
    ]
