import FWCore.ParameterSet.Config as cms

from SimG4CMS.Calo.hgcalHitIdCheckEE_cfi import *

hgcalHitIdCheckHEF = hgcalHitIdCheckEE.clone(
    nameDevice = "HGCal HE Silicon",
    nameSense  = "HGCalHESiliconSensitive",
    caloHitSource = "HGCHitsHEfront")

hgcalHitIdCheckHEB = hgcalHitIdCheckEE.clone(
    nameDevice = "HGCal HE Scinitillator",
    nameSense  = "HGCalHEScintillatorSensitive",
    caloHitSource = "HGCHitsHEback")
