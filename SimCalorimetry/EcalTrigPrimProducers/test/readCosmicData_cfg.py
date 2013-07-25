import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSEMIP")
# ECAL Unpacker
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

# ECAL TPG Producer
process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_CosmicsConfiguration_cff")

# ECAL TPG Analyzer
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/p/paganini/Data/ecal_local.00038343.0001.A.storageManager.0.0000.dat')
)

process.tpAnalyzer = cms.EDAnalyzer("EcalTrigPrimAnalyzerMIPs",
    Producer = cms.string('EBTT'),
    EmulProducer = cms.string(''),
    DigiLabel = cms.string('ecalEBunpacker'),
    DigiProducer = cms.string('ebDigis'),
    EmulLabel = cms.string('simEcalTriggerPrimitiveDigis'),
    Label = cms.string('ecalEBunpacker')
)

process.p = cms.Path(process.ecalEBunpacker*process.simEcalTriggerPrimitiveDigis*process.tpAnalyzer)


