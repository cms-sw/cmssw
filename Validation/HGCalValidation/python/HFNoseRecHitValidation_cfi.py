import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitValidationEE_cfi import *

hfnoseRecHitValidation = hgcalRecHitValidationEE.clone(
    DetectorName = cms.string("HGCalHFNoseSensitive"),
    RecHitSource = cms.InputTag("HGCalRecHit","HGCHFNoseRecHits"))

# foo bar baz
# RDa6q7ALrn92O
# Hj31Uf8Fd1YVL
