import FWCore.ParameterSet.Config as cms

hgcHitAnalysis = cms.EDAnalyzer("HGCHitValidation",
                                geometrySource = cms.untracked.vstring("HGCalEESensitive",
                                                                       "HGCalHESiliconSensitive",
                                                                       "Hcal"),
                                eeSimHitSource = cms.InputTag("g4SimHits","HGCHitsEE"),
                                fhSimHitSource = cms.InputTag("g4SimHits","HGCHitsHEfront"),
                                bhSimHitSource = cms.InputTag("g4SimHits","HcalHits"),
                                eeRecHitSource = cms.InputTag("HGCalRecHit","HGCEERecHits"),
                                fhRecHitSource = cms.InputTag("HGCalRecHit","HGCHEFRecHits"),
                                bhRecHitSource = cms.InputTag("HGCalRecHit","HGCHEBRecHits"),
#                                bhRecHitSource = cms.InputTag("hbhereco"),
                                ietaExcludeBH  = cms.vint32([]),
                                ifHCAL         = cms.bool(False)
                                )
