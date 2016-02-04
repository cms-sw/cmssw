# The following comments couldn't be translated into the new config version:

# Config file description:
#  test ECAL sequence: tp digi -> tcc hw input -> tp digi
#  check consistency of input and output tp digi collections

import FWCore.ParameterSet.Config as cms

process = cms.Process("TCCDigi2Digi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

#-#-# ECAL TPG
process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi")

#-#-# Digi -> Flat
process.load("SimCalorimetry.EcalElectronicsEmulation.EcalSimRawData_cfi")

#-#-# Flat -> Digi
process.load("SimCalorimetry.EcalElectronicsEmulation.EcalFEtoDigi_cfi")

#comparator
process.load("L1Trigger.HardwareValidation.L1Comparator_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:file_mc.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.EcalTrigPrimESProducer = cms.ESProducer("EcalTrigPrimESProducer",
    DatabaseFileEE = cms.untracked.string('TPG_EE.txt'),
    DatabaseFileEB = cms.untracked.string('TPG_EB.txt')
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

process.eegeom = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalMappingRcd'),
    iovIsRunNotTime = cms.bool(False),
    firstValid = cms.vuint32(1)
)

process.outputEvents = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ecalDigi2Digi.root')
)

process.p = cms.Path(process.ecalTriggerPrimitiveDigis*process.ecalSimRawData*process.tccFlatToDigi*process.l1Compare)
process.outpath = cms.EndPath(process.outputEvents)
process.ecalTriggerPrimitiveDigis.TcpOutput = True
process.ecalTriggerPrimitiveDigis.BarrelOnly = True
process.ecalTriggerPrimitiveDigis.Label = 'ecalDigis'
process.ecalTriggerPrimitiveDigis.InstanceEB = 'ebDigis'
process.ecalTriggerPrimitiveDigis.InstanceEE = 'eeDigis'
process.ecalTriggerPrimitiveDigis.Debug = False
process.ecalSimRawData.tcc2dccData = False
process.ecalSimRawData.srp2dccData = False
process.ecalSimRawData.fe2dccData = False
process.ecalSimRawData.trigPrimProducer = 'ecalTriggerPrimitiveDigis'
process.ecalSimRawData.tcpDigiCollection = 'formatTCP'
process.ecalSimRawData.tpVerbose = False
process.ecalSimRawData.tccInDefaultVal = 0
process.ecalSimRawData.tccNum = -1
process.ecalSimRawData.outputBaseName = 'data/ecal'
process.tccFlatToDigi.SuperModuleId = -1
process.tccFlatToDigi.FlatBaseName = 'data/ecal_tcc'
process.tccFlatToDigi.FileEventOffset = 0
process.l1Compare.ETP_dataLabel = 'tccFlatToDigi'
process.l1Compare.ETP_emulLabel = 'ecalTriggerPrimitiveDigis'
process.l1Compare.DumpFile = 'dump_ecal.txt'
process.l1Compare.DumpMode = 1
process.l1Compare.COMPARE_COLLS = [1, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 
    0]

