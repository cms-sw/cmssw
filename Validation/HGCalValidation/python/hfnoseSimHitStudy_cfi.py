import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalSimHitStudy_cfi import *

hgcalSimHitStudy.detectorNames  = cms.vstring('HGCalHFNoseSensitive')
hgcalSimHitStudy.caloHitSources = cms.vstring('HFNoseHits')
hgcalSimHitStudy.rMin    = cms.untracked.double(0)
hgcalSimHitStudy.rMax    = cms.untracked.double(1500)
hgcalSimHitStudy.zMin    = cms.untracked.double(10000)
hgcalSimHitStudy.zMax    = cms.untracked.double(11000)
hgcalSimHitStudy.etaMin  = cms.untracked.double(2.5)
hgcalSimHitStudy.etaMax  = cms.untracked.double(5.5)
hgcalSimHitStudy.nBinR   = cms.untracked.int32(150)
hgcalSimHitStudy.nBinZ   = cms.untracked.int32(100)
hgcalSimHitStudy.nBinEta = cms.untracked.int32(150)
