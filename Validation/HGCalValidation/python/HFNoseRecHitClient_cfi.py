import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalRecHitsClient_cfi import *

hfnoseRecHitClient = hgcalRecHitClientEE.clone(
    DetectorName  = cms.string("HGCalHFNoseSensitive"))
