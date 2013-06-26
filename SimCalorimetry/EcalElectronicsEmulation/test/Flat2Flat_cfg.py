# The following comments couldn't be translated into the new config version:

# Config file description:
#  test ECAL sequence: tcc hw input -> tp digi -> tcc hw input
#  check consistency of original and created tcc hardware input files

import FWCore.ParameterSet.Config as cms

process = cms.Process("TCCFlat2Flat")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#-#-# Flat -> Digi  [create digis from input (set of) flat file(s)]
process.load("SimCalorimetry.EcalElectronicsEmulation.EcalFEtoDigi_cfi")

#-#-# Digi -> Flat
process.load("SimCalorimetry.EcalElectronicsEmulation.EcalSimRawData_cfi")

#dump digi collections
process.load("L1Trigger.HardwareValidation.L1Comparator_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2048)
)
process.source = cms.Source("EmptySource")

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

process.EcalTrigPrimESProducer = cms.ESProducer("EcalTrigPrimESProducer",
    DatabaseFileEE = cms.untracked.string('TPG_EE.txt'),
    #untracked string DatabaseFileEB = "TPG_poweron.txt"//identity tcc lut
    DatabaseFileEB = cms.untracked.string('TPG_EB.txt')
)

process.outputEvents = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ecalFlat2Flat.root')
)

process.p = cms.Path(process.tccFlatToDigi*process.ecalSimRawData*process.l1Compare)
process.outpath = cms.EndPath(process.outputEvents)
process.tccFlatToDigi.SuperModuleId = -1
process.tccFlatToDigi.FlatBaseName = 'data_in/ecal_tcc'
process.tccFlatToDigi.FileEventOffset = 0
process.tccFlatToDigi.UseIdentityLUT = False
process.tccFlatToDigi.debugPrintFlag = False
process.ecalSimRawData.tcc2dccData = False
process.ecalSimRawData.srp2dccData = False
process.ecalSimRawData.fe2dccData = False
process.ecalSimRawData.trigPrimProducer = 'tccFlatToDigi'
process.ecalSimRawData.tcpDigiCollection = 'formatTCP'
process.ecalSimRawData.tpVerbose = False
process.ecalSimRawData.tccInDefaultVal = 0
process.ecalSimRawData.tccNum = -1
process.ecalSimRawData.outputBaseName = 'data_out/ecal'
process.l1Compare.ETP_dataLabel = 'tccFlatToDigi'
process.l1Compare.ETP_emulLabel = 'tccFlatToDigi'
process.l1Compare.DumpFile = 'dump_flat.txt'
process.l1Compare.DumpMode = 1
process.l1Compare.COMPARE_COLLS = [1, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 
    0]

