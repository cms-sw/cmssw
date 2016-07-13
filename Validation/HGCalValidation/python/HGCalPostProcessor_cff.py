import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.HGCalSimHitsClientV6_cff import *
from Validation.HGCalValidation.HGCalDigiClientV6_cff    import *
from Validation.HGCalValidation.HGCalRecHitsClientV6_cff import *
from Validation.HGCalValidation.hgcGeometryClient_cfi    import *

hgcalPostProcessor = cms.Sequence(hgcalGeometryClient+hgcalSimHitClientEE+hgcalSimHitClientHEF+hgcalSimHitClientHEB+hgcalDigiClientEE+hgcalDigiClientHEF+hgcalDigiClientHEB+hgcalRecHitClientEE+hgcalRecHitClientHEF+hgcalRecHitClientHEB)
