import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiStudyEE_cfi import *

hgcalDigiStudyHEF = hgcalDigiStudyEE.clone(
    detectorName = cms.string("HGCalHESiliconSensitive"),
    digiSource   = cms.InputTag("hgcalDigis","HEfront"))

hgcalDigiStudyHEB = hgcalDigiStudyEE.clone(
    detectorName = cms.string("HCal"),
    digiSource   = cms.InputTag("hgcalDigis","HEback"))

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalDigiStudyHEB,
                        detectorName = cms.string("HGCalHEScintillatorSensitive")
                        )
