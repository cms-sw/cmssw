import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalDigiClient_cfi import *

hgcalDigiClientHEF = hgcalDigiClientEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"))

hgcalDigiClientHEB = hgcalDigiClientEE.clone(
    DetectorName  = cms.string("HCal"))

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalDigiClientHEB,
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"),
)
