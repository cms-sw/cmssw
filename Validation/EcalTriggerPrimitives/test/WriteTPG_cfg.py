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

process.load("SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_with_suppressed_cff")


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_3_0/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/F6794108-E5BC-DE11-8639-0018F3D096DE.root',
       '/store/relval/CMSSW_3_3_0/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/EED7DC04-4CBD-DE11-BC61-002354EF3BE0.root',
       '/store/relval/CMSSW_3_3_0/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/82484FAB-E3BC-DE11-B79C-001731AF67E9.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *_*_*_*', 
	'keep *_simEcalTriggerPrimitiveDigis_*_*'),
    fileName = cms.untracked.string('file:TrigPrim_330rel.root')
)

process.Timing = cms.Service("Timing")

process.MessageLogger = cms.Service("MessageLogger")

process.p = cms.Path(process.simEcalTriggerPrimitiveDigis)
process.outpath = cms.EndPath(process.out)


