import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalWaferHitCheckEE_cfi import *

hgcalWaferHitCheckHEF = hgcalWaferHitCheckEE.clone(
    detectorName = cms.string("HGCalHESiliconSensitive"),
    caloHitSource = cms.string('HGCHitsHEfront'),
    source   = cms.InputTag("simHGCalUnsuppressedDigis","HEfront"))
# foo bar baz
# X291gziKUlNDm
# Mnn03GXK1R1gp
