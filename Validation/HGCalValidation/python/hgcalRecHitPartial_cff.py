import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitPartialEE_cfi import *

hgcalRecHitPartialHE = hgcalRecHitPartialEE.clone(
    detectorName = 'HGCalHESiliconSensitive',
    source = 'HGCalRecHit::HGCHEFRecHits'
)
