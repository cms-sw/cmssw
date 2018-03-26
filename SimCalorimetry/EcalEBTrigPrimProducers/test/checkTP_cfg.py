import FWCore.ParameterSet.Config as cms

process = cms.Process("PROdTPA")


process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'94X_mc2017_realistic_v10', '')


process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibCalorimetry.EcalTPGTools.ecalTPGScale_cff")

process.source = cms.Source("PoolSource",

fileNames = cms.untracked.vstring('file:EBTP_PhaseII_clu3x5_NewTEST.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff import *

process.tpAnalyzer = cms.EDAnalyzer("EcalEBTrigPrimAnalyzer",
                                    outFileName = cms.string('histos_clu3x5_GT1GeV_NewTEST.root'),
                                    inputRecHitsEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
                                    AnalyzeRecHits = cms.bool(True),
                                    AnalyzeElectrons = cms.bool(True),
                                    RecoContentAvailable =  cms.bool(True),
                                    Debug = cms.bool(False),
#                                    inputTP = cms.InputTag("EcalEBTrigPrimProducer")
                                    inputTP = cms.InputTag("simEcalEBTriggerPrimitiveDigis"),
                                    inputClusterTP = cms.InputTag("simEcalEBClusterTriggerPrimitiveDigis"),
                                    etCluTPThreshold = cms.double (1.),
                                    bxInfos = cms.InputTag("addPileupInfo"),
                                    genParticles = cms.InputTag("genParticles"),
                                    eleCollection = cms.InputTag("gedGsfElectrons"),
                                    phoCollection = cms.InputTag("gedPhotons")            
)



process.p = cms.Path(process.tpAnalyzer)
