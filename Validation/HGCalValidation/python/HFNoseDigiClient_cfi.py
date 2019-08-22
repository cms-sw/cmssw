import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalDigiClient_cfi import *

hfnoseDigiClient = hgcalDigiClientEE.clone(
    DetectorName  = cms.string("HGCalHFNoseSensitive"))
