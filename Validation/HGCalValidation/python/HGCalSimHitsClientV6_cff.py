import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalSimHitsClient_cfi import *

hgcalSimHitClientHEF = hgcalSimHitClientEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"))

hgcalSimHitClientHEB = hgcalSimHitClientEE.clone(
    DetectorName  = cms.string("HCal"))
