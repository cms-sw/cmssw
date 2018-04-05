import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalSimHitsClient_cff import *
from Validation.HGCalValidation.HGCalDigiClient_cff    import *
from Validation.HGCalValidation.HGCalRecHitsClient_cff import *

hgcalPostProcessor = cms.Sequence(hgcalSimHitClientEE+hgcalSimHitClientHEF+hgcalSimHitClientHEB+hgcalDigiClientEE+hgcalDigiClientHEF+hgcalDigiClientHEB+hgcalRecHitClientEE+hgcalRecHitClientHEF+hgcalRecHitClientHEB)
