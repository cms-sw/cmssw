import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalSimHitsClient_cff import *
from Validation.HGCalValidation.HGCalDigiClient_cff    import *
from Validation.HGCalValidation.HGCalRecHitsClient_cff import *
from Validation.HGCalValidation.PostProcessorHGCAL_cfi import postProcessorHGCAL

hgcalPostProcessor = cms.Sequence(hgcalSimHitClientEE
    + hgcalSimHitClientHEF
    + hgcalSimHitClientHEB
    + hgcalDigiClientEE
    + hgcalDigiClientHEF
    + hgcalDigiClientHEB
    + hgcalRecHitClientEE
    + hgcalRecHitClientHEF
    + hgcalRecHitClientHEB)

hgcalValidatorPostProcessor = cms.Sequence(postProcessorHGCAL)
