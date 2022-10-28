import FWCore.ParameterSet.Config as cms

process = cms.Process("PROdTPA")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibCalorimetry.EcalTPGTools.ecalTPGScale_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:TrigPrim.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.tpAnalyzer = cms.EDAnalyzer("EcalTrigPrimAnalyzer",
    inputRecHitsEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    inputRecHitsEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    AnalyzeRecHits = cms.bool(False),
    inputTP = cms.InputTag("simEcalTriggerPrimitiveDigis","","PROTPGD")
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('histos.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.p = cms.Path(process.tpAnalyzer)


