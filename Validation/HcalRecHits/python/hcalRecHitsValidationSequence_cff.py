import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitParam_cfi import *
import Validation.HcalRecHits.HcalRecHitParam_cfi
HBRecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
import Validation.HcalRecHits.HcalRecHitParam_cfi
HERecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
import Validation.HcalRecHits.HcalRecHitParam_cfi
HORecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
import Validation.HcalRecHits.HcalRecHitParam_cfi
HFRecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
import Validation.HcalRecHits.HcalRecHitParam_cfi
AllRecHitsValidation = Validation.HcalRecHits.HcalRecHitParam_cfi.hcalRecoAnalyzer.clone()
hcalRecHitsValidationSequence = cms.Sequence(HBRecHitsValidation+HERecHitsValidation+HORecHitsValidation+HFRecHitsValidation+AllRecHitsValidation)
HBRecHitsValidation.hcalselector = 'HB'
HBRecHitsValidation.outputFile = 'HcalRecHitsValidationHB.root'
HERecHitsValidation.hcalselector = 'HE'
HERecHitsValidation.outputFile = 'HcalRecHitsValidationHE.root'
HORecHitsValidation.hcalselector = 'HO'
HORecHitsValidation.outputFile = 'HcalRecHitsValidationHO.root'
HFRecHitsValidation.hcalselector = 'HF'
HFRecHitsValidation.outputFile = 'HcalRecHitsValidationHF.root'
AllRecHitsValidation.hcalselector = 'all'
AllRecHitsValidation.outputFile = 'HcalRecHitsValidationAll.root'


