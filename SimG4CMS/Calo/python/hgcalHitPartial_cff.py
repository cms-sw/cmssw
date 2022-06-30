import FWCore.ParameterSet.Config as cms

from SimG4CMS.Calo.hgcalHitPartialEE_cfi import *

hgcalHitPartialHE = hgcalHitPartialEE.clone(
    nameSense  = cms.string("HGCalHESiliconSensitive"),
    caloHitSource = cms.string("HGCHitsHEfront"))
