import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSelectiveReadoutValid")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# initialize magnetic field
process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

# geometry (Only Ecal)
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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/CRUZET2/Cosmics/RAW/v1/000/046/629/9A33ABCA-F037-DD11-8D85-000423D6BA18.root', 
        '/store/data/CRUZET2/Cosmics/RAW/v1/000/046/629/9CE2CDB9-F137-DD11-9840-000423D992DC.root', 
        '/store/data/CRUZET2/Cosmics/RAW/v1/000/046/629/F46E124B-F137-DD11-9FF4-000423D6B358.root')
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.tpparams12 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPhysicsConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(10)
)

process.p1 = cms.Path(process.ecalEBunpacker*process.ecalSelectiveReadoutValidation)
process.MessageLogger.cerr.INFO.limit = 10
process.MessageLogger.suppressWarning = ['ecalSelectiveReadoutValidation']
process.DQM.collectorHost = ''
process.ecalSelectiveReadoutValidation.EbDigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalSelectiveReadoutValidation.EeDigiCollection = 'ecalEBunpacker:eeDigis'
process.ecalSelectiveReadoutValidation.EbSrFlagCollection = 'ecalEBunpacker:'
process.ecalSelectiveReadoutValidation.EeSrFlagCollection = 'ecalEBunpacker:'
process.ecalSelectiveReadoutValidation.TrigPrimCollection = 'ecalEBunpacker:EcalTriggerPrimitives'
process.ecalSelectiveReadoutValidation.tpInGeV = False
process.ecalSelectiveReadoutValidation.histDir = ''
process.ecalSelectiveReadoutValidation.histograms = ['dccVolFromData', #DCC event fragment size                    
   'forcedTt',       #Trigger tower readout forced bit on
   'fullRoTt',       #Full Read-out trigger tower               
   'hiTtf',          #Low interest trigger tower flags 1 distribution
   'liTtf',          #Low interest trigger tower flags iPhi   
   'vol',            #ECAL data volume             
   'volB',           #Barrel data volume
   'volBHI',         #Barrel high interest data volume
   'volBLI',         #Barrel low interest data volume                 
   'volE',           #Endcap data volume                  
   'volEHI',         #Endcap high interest data volume
   'volELI',         #Endcap low interest data volume                 
   'volHI'           #ECAL high interest data volume
  ]

