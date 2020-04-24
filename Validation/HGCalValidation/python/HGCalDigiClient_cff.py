import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalDigiClient_cfi import *

hgcalDigiClientHEF = hgcalDigiClientEE.clone(
    DetectorName  = cms.string("HGCalHESiliconSensitive"))

hgcalDigiClientHEB = hgcalDigiClientEE.clone(
    DetectorName  = cms.string("HCal"))
