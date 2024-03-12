import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitStudyEE_cfi import *

hgcalRecHitStudyFH = hgcalRecHitStudyEE.clone(
    detectorName  = cms.string("HGCalHESiliconSensitive"),
    source        = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"),
    layers        = cms.untracked.int32(24))

hgcalRecHitStudyBH = hgcalRecHitStudyEE.clone(
    detectorName  = cms.string("HGCalHEScintillatorSensitive"),
    source        = cms.InputTag("HGCalRecHit", "HGCHEBRecHits"),
    layers        = cms.untracked.int32(24))
# foo bar baz
# uoRTw6m9kXTVC
# jwbo8p9mh650a
