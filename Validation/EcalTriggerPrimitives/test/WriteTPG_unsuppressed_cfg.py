import FWCore.ParameterSet.Config as cms

process = cms.Process("PROTPGD")
# ecal mapping
process.load("Geometry.EcalMapping.EcalMapping_cfi")

process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

# IdealGeometryRecord
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibCalorimetry.Configuration.Ecal_FakeConditions_cff")

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff")


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/data/uberthon/tpg/elec_unsupp_pt10-100.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
	'keep *_simEcalTriggerPrimitiveDigis_*_*'),
    fileName = cms.untracked.string('file:TrigPrim_unsuppressed.root')
)

process.Timing = cms.Service("Timing")

process.MessageLogger = cms.Service("MessageLogger")

process.p = cms.Path(process.simEcalTriggerPrimitiveDigis)
process.outpath = cms.EndPath(process.out)


