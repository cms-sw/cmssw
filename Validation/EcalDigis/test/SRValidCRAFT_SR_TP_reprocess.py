import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSelectiveReadoutValid")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('to_be_replaced')
)

import FWCore.Modules.printContent_cfi
process.dumpEv = FWCore.Modules.printContent_cfi.printContent.clone()

# MessageLogger:
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10
process.MessageLogger.suppressWarning = ['ecalSelectiveReadoutValidation']

# DQM services
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

# ECAL Unpacker:
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.ecalEBunpacker.silentMode = cms.untracked.bool(True)


# ECAL Geometry:
process.load("Geometry.EcalCommonData.EcalOnly_cfi")
#  Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

#Magnetic field:
#process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")
#process.load("Configuration.StandardSequences.FakeConditions_cff")


process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

#List of collection
#process.load("pgras.ListCollection.ListCollection_cfi")


# SR emulation:
process.load("SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_craft_cfi")
process.simEcalDigis.trigPrimProducer = "ecalTriggerPrimitiveDigis"
process.simEcalDigis.digiProducer = cms.string('ecalEBunpacker')
process.simEcalDigis.EBdigiCollection = cms.string('ebDigis')
process.simEcalDigis.EEdigiCollection = cms.string('eeDigis')

# Srp validation (analysis module):
#   Defines Ecal seletive readout validation module, ecalSelectiveReadoutValidation:
process.load("Validation.EcalDigis.ecalSelectiveReadoutValidation_cfi")
process.ecalSelectiveReadoutValidation.EbDigiCollection = 'simEcalDigis:ebDigis'
process.ecalSelectiveReadoutValidation.EeDigiCollection = 'simEcalDigis:eeDigis'
process.ecalSelectiveReadoutValidation.EbSrFlagCollection = 'simEcalDigis:ebSrFlags'
process.ecalSelectiveReadoutValidation.EeSrFlagCollection = 'simEcalDigis:eeSrFlags'
process.ecalSelectiveReadoutValidation.TrigPrimCollection = 'simEcalTriggerPrimitiveDigis'

#process.ecalSelectiveReadoutValidation.EbDigiCollection = 'ecalEBunpacker:ebDigis'
#process.ecalSelectiveReadoutValidation.EeDigiCollection = 'ecalEBunpacker:eeDigis'
#process.ecalSelectiveReadoutValidation.EbSrFlagCollection = 'ecalEBunpacker:'
#process.ecalSelectiveReadoutValidation.EeSrFlagCollection = 'ecalEBunpacker:'
#process.ecalSelectiveReadoutValidation.TrigPrimCollection = 'ecalEBunpacker:EcalTriggerPrimitives'

process.ecalSelectiveReadoutValidation.tpInGeV = False
process.ecalSelectiveReadoutValidation.ecalDccZs1stSample = 3
process.ecalSelectiveReadoutValidation.dccWeights = [ -1.1865, 0.0195, 0.2900, 0.3477, 0.3008, 0.2266 ]
process.ecalSelectiveReadoutValidation.histDir = ''
process.ecalSelectiveReadoutValidation.histograms = [
    "EBEventSize",           #Barrel data volume
    "EBHighInterestPayload", #Barrel high interest crystal data payload
    "EBLowInterestPayload",  #ECAL Barrel low interest crystal data payload
    "EcalMidInterestTriggerTowerFlagMap", #Mid interest trigger tower flags
    "EEEventSize",           #Endcap data volume
    "EEHighInterestPayload", #Endcap high interest crystal data payload
    "EELowInterestPayload",  #Endcap low interest crystal data payload
    "EbZeroSupp1SRFlagMap",  #Trigger tower read-out with ZS threshold 1
    "EcalChannelOccupancy",  #ECAL crystal channel occupancy after zero suppression
    "EcalDccEventSize",      #ECAL DCC event fragment size
    "EcalDccEventSizeComputed", #ECAL DCC event fragment size
    "EcalEventSize",         #ECAL data volume
    "EcalFullReadoutSRFlagMap", #Full Read-out trigger tower
    "EcalHighInterestPayload",  #ECAL high interest crystal data payload
    "EcalHighInterestTriggerTowerFlagMap", #High interest trigger tower flags
    "EcalLowInterestPayload",#ECAL low interest crystal data payload
    "EcalLowInterestTriggerTowerFlagMap", #Low interest trigger tower flags
    "EcalMidInterestTriggerTowerFlagMap", #Mid interest trigger tower flags
    "EcalReadoutUnitForcedBitMap", #ECAL readout unit with forced bit of SR flag on
    "EcalTriggerPrimitiveEt",#Trigger primitive TT E_{T}
    "EcalTriggerPrimitiveEtMap", #Trigger primitive
    "EcalTriggerTowerFlag",  #Trigger primitive TT flag
    "hEbEMean",              #EE <E_hit>
    "hEbNoZsRecVsSimE",      #Crystal no-zs simulated vs reconstructed energy
    "hEbRecE",               #Crystal reconstructed energy
    "hEbRecEHitXtal",        #EB rec energy of hit crystals
    "hEeEMean",              #EE <E_hit>
    "hEeRecE",               #EE crystal reconstructed energy
    "tpVsEtSum",             #Trigger primitive Et (TP) vs #sumE_{T}
    "ttfVsEtSum",            #Trigger tower flag vs #sumE_{T}
    "ttfVsTp",               #Trigger tower flag vs TP
    "zsEbHiFIRemu",          #Emulated ouput of ZS FIR filter for EB high interest crystals
    "zsEbLiFIRemu",          #Emulated ouput of ZS FIR filter for EB low interest crystals
    "zsEeHiFIRemu",          #Emulated ouput of ZS FIR filter for EE high interest crystals
    "zsEeLiFIRemu"           #Emulated ouput of ZS FIR filter for EE low interest crystals
    ]
process.tpparams = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLinearizationConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams2 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPedestalsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams3 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGSlidingWindowRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams4 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGWeightIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams5 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGWeightGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams6 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLutGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams7 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLutIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams8 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainEBIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams9 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainEBGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams10 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainStripEERcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams11 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainTowerEERcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.EcalTrigPrimESProducer = cms.ESProducer("EcalTrigPrimESProducer",
    DatabaseFile = cms.untracked.string('TPG_startup.txt.gz')
)

process.ecalTriggerPrimitiveDigis = cms.EDProducer("EcalTrigPrimProducer",
    InstanceEB = cms.string('ebDigis'),
    InstanceEE = cms.string('eeDigis'),
    Label = cms.string('ecalEBunpacker'),

    BarrelOnly = cms.bool(False),
    Famos = cms.bool(False),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),

    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
                                                   
    TTFHighEnergyEB = cms.double(1.0),
    TTFHighEnergyEE = cms.double(1.0),
    TTFLowEnergyEB = cms.double(1.0), ## this + the following is added from 140_pre4 on
    TTFLowEnergyEE = cms.double(1.0)
)


#process.tpAnalyzer = cms.EDAnalyzer("EcalTPGAnalyzer",
#
#    Label = cms.string('ecalEBunpacker'),
#    Producer = cms.string('EcalTriggerPrimitives'),
#    DigiLabel = cms.string('ecalEBunpacker'),
#    DigiProducerEE = cms.string(''),
#    DigiProducerEB = cms.string('ebDigis'),
#    EmulLabel = cms.string('ecalTriggerPrimitiveDigis'),
#    EmulProducer = cms.string(''),
#
#    Print = cms.bool(True),
#    CompareWithPreviousEvent = cms.bool(True),
#    ReadTriggerPrimitives = cms.bool(True),                                    
#    UseEndCap = cms.bool(False),
#    KeepOnlyL1Ecal = cms.bool(True) ,
#
#    ADCCut = cms.int32(3), ## crystal energy contributes to tower energy if above the threshold
#    shapeCut = cms.int32(3), ## pulse profile filled with tower energy if above the threshold
#    occupancyCut = cms.int32(3), ## occupancy plots filled if tpg above the threshold
#
#    TPGEmulatorIndexRef = cms.int32(2) ## must be in [0,4]
#)
#

process.p = cms.Path(process.ecalEBunpacker*process.ecalTriggerPrimitiveDigis*process.simEcalDigis*process.ecalSelectiveReadoutValidation)

process.ecalSelectiveReadoutValidation.outputFile = 'run69912hists_.root'

process.source.fileNames = ['/store/data/Commissioning08/Cosmics/RAW/v1/000/069/912/049C2F4D-10AD-DD11-BFEA-000423D174FE.root' ]
