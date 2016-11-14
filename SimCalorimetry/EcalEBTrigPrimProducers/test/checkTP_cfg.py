import FWCore.ParameterSet.Config as cms

process = cms.Process("PROdTPA")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibCalorimetry.EcalTPGTools.ecalTPGScale_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:EBTP_PhaseII_RelVal2017HGGWithPU.root')
#                            fileNames = cms.untracked.vstring('file:EBTP_PhaseII.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff import *

process.tpAnalyzer = cms.EDAnalyzer("EcalEBTrigPrimAnalyzer",
                                    inputRecHitsEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
                                    AnalyzeRecHits = cms.bool(True),
                                    Debug = cms.bool(True),
                                    inputTP = cms.InputTag("EcalEBTrigPrimProducer")
    
)



process.p = cms.Path(process.tpAnalyzer)
