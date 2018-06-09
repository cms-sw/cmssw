import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hgcalHitValidation = DQMEDAnalyzer('HGCalHitValidation',
                                    geometrySource = cms.untracked.vstring("HGCalEESensitive",
                                                                           "HGCalHESiliconSensitive",
                                                                           "Hcal"),
                                    eeSimHitSource = cms.InputTag("g4SimHits","HGCHitsEE"),
                                    fhSimHitSource = cms.InputTag("g4SimHits","HGCHitsHEfront"),
                                    bhSimHitSource = cms.InputTag("g4SimHits","HcalHits"),
                                    eeRecHitSource = cms.InputTag("HGCalRecHit","HGCEERecHits"),
                                    fhRecHitSource = cms.InputTag("HGCalRecHit","HGCHEFRecHits"),
								    bhRecHitSource = cms.InputTag("HGCalRecHit","HGCHEBRecHits"),
                                    ietaExcludeBH  = cms.vint32([]),
                                    ifHCAL         = cms.bool(False),
                                    ifHCALsim      = cms.bool(True),
                                    )

from Validation.HGCalValidation.hgcalHitCalibration_cfi import hgcalHitCalibration
from Validation.HGCalValidation.caloparticlevalidation_cfi import caloparticlevalidation

hgcalHitValidationSequence = cms.Sequence(hgcalHitValidation+hgcalHitCalibration+caloparticlevalidation)

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
phase2_hgcalV9.toModify(hgcalHitValidation,
    bhSimHitSource = cms.InputTag("g4SimHits","HGCHitsHEback"),
    geometrySource = cms.untracked.vstring("HGCalEESensitive","HGCalHESiliconSensitive","HGCalHEScintillatorSensitive"),
    ifHCALsim      = cms.bool(False),
)
