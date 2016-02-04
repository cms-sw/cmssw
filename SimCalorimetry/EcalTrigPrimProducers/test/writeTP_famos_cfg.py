import FWCore.ParameterSet.Config as cms

process = cms.Process("PROTPGD")
# ecal mapping
process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

# magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

# IdealGeometryRecord
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/uberthon/tpg2/famos_input2.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:TrigPrim_famos.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.ProfilerService = cms.Service("ProfilerService",
    lastEvent = cms.untracked.int32(10),
    firstEvent = cms.untracked.int32(1),
    paths = cms.untracked.vstring('p')
)

process.p = cms.Path(process.simEcalTriggerPrimitiveDigis)
process.outpath = cms.EndPath(process.out)
process.simEcalTriggerPrimitiveDigis.Famos = True
process.simEcalTriggerPrimitiveDigis.Label = 'RecHits'


