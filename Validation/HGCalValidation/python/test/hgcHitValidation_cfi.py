import FWCore.ParameterSet.Config as cms

hgcHitAnalysis = cms.EDAnalyzer("HGCHitValidation",
                                geometrySource = cms.untracked.vstring("HGCalEESensitive",
                                                                       "HGCalHESiliconSensitive",
                                                                       "Hcal"),
                                eeSimHitSource = cms.InputTag("g4SimHits","HGCHitsEE"),
                                fhSimHitSource = cms.InputTag("g4SimHits","HGCHitsHEfront"),
                                bhSimHitSource = cms.InputTag("g4SimHits","HcalHits"),
                                eeUncalibRecHitSource = cms.InputTag("HGCalUncalibRecHit","HGCEEUncalibRecHits"),
                                fhUncalibRecHitSource = cms.InputTag("HGCalUncalibRecHit","HGCHEFUncalibRecHits"),
                                bhUncalibRecHitSource = cms.InputTag("hbheUpgradeReco")
                                )
