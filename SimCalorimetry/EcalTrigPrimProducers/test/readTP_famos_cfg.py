import FWCore.ParameterSet.Config as cms

process = cms.Process("PROdTPA")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/MyFirstFamosFile.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.tpAnalyzer = cms.EDAnalyzer("EcalTrigPrimAnalyzer",
    RecHitsProducer = cms.string('ecalRecHit'),
    Producer = cms.string(''),
    RecHitsLabelEB = cms.string('EcalRecHitsEB'),
    RecHitsLabelEE = cms.string('EcalRecHitsEE'),
    Label = cms.string('simEcalTriggerPrimitiveDigis'),
    AnalyzeRecHits = cms.bool(False)
)

process.p = cms.Path(process.tpAnalyzer)


