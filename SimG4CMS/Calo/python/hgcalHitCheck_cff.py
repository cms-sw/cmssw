import FWCore.ParameterSet.Config as cms

from SimG4CMS.Calo.hgcalHitCheckEE_cfi import *
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify( hgcalHitCheckEE, tag = "DD4hep" )

hgcalHitCheckHEF = hgcalHitCheckEE.clone(
    nameDevice = "HGCal HE Silicon",
    nameSense  = "HGCalHESiliconSensitive",
    caloHitSource = "HGCHitsHEfront",
    layers = 21)

hgcalHitCheckHEB = hgcalHitCheckEE.clone(
    nameDevice = "HGCal HE Scinitillator",
    nameSense  = "HGCalHEScintillatorSensitive",
    caloHitSource = "HGCHitsHEback",
    layers = 21)
