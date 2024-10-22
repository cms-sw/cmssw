import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalDigiStudyEE_cfi import *

hfnoseDigiStudy = hgcalDigiStudyEE.clone(
    detectorName = "HGCalHFNoseSensitive",
    digiSource   = "hfnoseDigis : HFNose",
    ifNose       = True,
    rMin         = 0,
    rMax         = 150,
    zMin         = 1000,
    zMax         = 1100,
    etaMin       = 2.5,
    etaMax       = 5.5,
    nBinR        = 150,
    nBinZ        = 100,
    nBinEta      = 150,
    layers       = 8,
    ifLayer      = True
    )
