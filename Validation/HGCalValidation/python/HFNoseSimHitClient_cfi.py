import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalSimHitsClient_cfi import *

hfnoseSimHitClient = hgcalSimHitClientEE.clone(
    DetectorName  = cms.string("HFNoseSensitive"))
