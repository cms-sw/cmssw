import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

hgcalHitValidation = DQMEDAnalyzer('HGCalHitValidation',
                                   geometrySource = cms.vstring("HGCalEESensitive",
                                                                "HGCalHESiliconSensitive",
                                                                "HGCalHEScintillatorSensitive"),
                                   eeSimHitSource = cms.InputTag("g4SimHits","HGCHitsEE"),
                                   fhSimHitSource = cms.InputTag("g4SimHits","HGCHitsHEfront"),
                                   bhSimHitSource = cms.InputTag("g4SimHits","HGCHitsHEback"),
                                   eeRecHitSource = cms.InputTag("HGCalRecHit","HGCEERecHits"),
                                   fhRecHitSource = cms.InputTag("HGCalRecHit","HGCHEFRecHits"),
                                   bhRecHitSource = cms.InputTag("HGCalRecHit","HGCHEBRecHits"),
                                   ietaExcludeBH  = cms.vint32([]),
)

from Validation.HGCalValidation.hgcalHitCalibration_cfi import hgcalHitCalibration
from Validation.HGCalValidation.caloparticlevalidation_cfi import caloparticlevalidation

hgcalHitValidationSequence = cms.Sequence(hgcalHitValidation+hgcalHitCalibration+caloparticlevalidation)
