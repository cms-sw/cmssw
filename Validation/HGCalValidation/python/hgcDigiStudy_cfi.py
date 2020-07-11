import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiStudyEE_cfi import *

hgcalDigiStudyHEF = hgcalDigiStudyEE.clone(
    detectorName = cms.string("HGCalHESiliconSensitive"),
    digiSource   = cms.InputTag("hgcalDigis","HEfront"),
    layers       = cms.untracked.int32(24))

hgcalDigiStudyHEB = hgcalDigiStudyEE.clone(
    detectorName = cms.string("HGCalHEScintillatorSensitive"),
    digiSource   = cms.InputTag("hgcalDigis","HEback"),
    layers       = cms.untracked.int32(24))
