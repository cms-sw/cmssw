import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalSiliconAnalysisEE_cfi import *

hgcalSiliconAnalysisHEF = hgcalSiliconAnalysisEE.clone(
    detectorName = cms.untracked.string("HGCalHESiliconSensitive"),
    HitCollection = cms.untracked.string('HGCHitsHEfront'),
    DigiCollection = cms.untracked.InputTag("simHGCalUnsuppressed","HEfront"))
