import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.hgcalRecHitStudyEE_cfi import *

hfnoseRecHitStudy = hgcalRecHitStudyEE.clone(
    detectorName = "HGCalHFNoseSensitive",
    source   = "HGCalRecHit : HGCHFNoseRecHits",
    ifNose   = True,
    rMin     = 0,
    rMax     = 150,
    zMin     = 1000,
    zMax     = 1100,
    etaMin   = 2.5,
    etaMax   = 5.5,
    nBinR    = 150,
    nBinZ    = 100,
    nBinEta  = 150,
    layers   = 8,
    ifLayer  = True
    )
