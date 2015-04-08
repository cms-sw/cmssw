import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalRecHitsClient_cfi import *

hgcalRecHitClientHEF = hgcalRecHitClientEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"))

hgcalRecHitClientHEB = hgcalRecHitClientEE.clone(
    DetectorName  = cms.string("HGCalHEScintillatorSensitive"))
