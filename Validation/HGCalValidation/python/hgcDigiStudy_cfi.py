import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiStudyEE_cfi import *

hgcalDigiStudyHEF = hgcalDigiStudyEE.clone(
    detectorName = cms.string("HGCalHESiliconSensitive"),
    digiSource   = cms.InputTag("simHGCalUnsuppressedDigis","HEfront"),
    layers       = cms.untracked.int32(24))

hgcalDigiStudyHEB = hgcalDigiStudyEE.clone(
    detectorName = cms.string("HGCalHEScintillatorSensitive"),
    digiSource   = cms.InputTag("simHGCalUnsuppressedDigis","HEback"),
    layers       = cms.untracked.int32(24))
