import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiStudyEE_cfi import *

hfnoseDigiStudy = hgcalDigiStudyEE.clone(
    detectorName = cms.string("HGCalHFNoseSensitive"),
    digiSource   = cms.InputTag("hfnoseDigis","HFNose"),
    ifNose       = cms.untracked.bool(True),
    rMin         = cms.untracked.double(0),
    rMax         = cms.untracked.double(150),
    zMin         = cms.untracked.double(1000),
    zMax         = cms.untracked.double(1100),
    etaMin       = cms.untracked.double(2.5),
    etaMax       = cms.untracked.double(5.5),
    nBinR        = cms.untracked.int32(150),
    nBinZ        = cms.untracked.int32(100),
    nBinEta      = cms.untracked.int32(150),
    layers       = cms.untracked.int32(8),
    ifLayer      = cms.untracked.bool(True)
    )
