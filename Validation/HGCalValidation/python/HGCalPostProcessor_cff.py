import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalSimHitsClient_cff import *
from Validation.HGCalValidation.HGCalDigiClient_cff    import *
from Validation.HGCalValidation.HGCalRecHitsClient_cff import *
from Validation.HGCalValidation.hgcGeometryClient_cfi  import *

hgcalPostProcessor = cms.Sequence(hgcalGeometryClient+hgcalSimHitClientEE+hgcalSimHitClientHEF+hgcalSimHitClientHEB+hgcalDigiClientEE+hgcalDigiClientHEF+hgcalDigiClientHEB+hgcalRecHitClientEE+hgcalRecHitClientHEF+hgcalRecHitClientHEB)
