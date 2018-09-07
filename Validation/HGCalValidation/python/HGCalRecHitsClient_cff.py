import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalRecHitsClient_cfi import *

hgcalRecHitClientHEF = hgcalRecHitClientEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"))

hgcalRecHitClientHEB = hgcalRecHitClientEE.clone(
    DetectorName  = cms.string("HCal"))

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify(hgcalRecHitClientHEB,
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"),
)
