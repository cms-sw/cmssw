import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalSimHitsClient_cfi import *

hgcalSimHitClientHEF = hgcalSimHitClientEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"))

hgcalSimHitClientHEB = hgcalSimHitClientEE.clone(
    DetectorName  = cms.string("HCal"))

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalSimHitClientHEB,
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"),
)
