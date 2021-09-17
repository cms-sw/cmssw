import FWCore.ParameterSet.Config as cms

from SimG4CMS.Calo.hgcalHitIdCheckEE_cfi import *

hgcalHitIdCheckHEF = hgcalHitIdCheckEE.clone(
    nameDevice = cms.string("HGCal HE Silicon"),
    nameSense  = cms.string("HGCalHESiliconSensitive"),
    caloHitSource = cms.string("HGCHitsHEfront"))

hgcalHitIdCheckHEB = hgcalHitIdCheckEE.clone(
    nameDevice = cms.string("HGCal HE Scinitillator"),
    nameSense  = cms.string("HGCalHEScintillatorSensitive"),
    caloHitSource = cms.string("HGCHitsHEback"))
