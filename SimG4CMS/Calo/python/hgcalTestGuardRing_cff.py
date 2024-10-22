import FWCore.ParameterSet.Config as cms

from SimG4CMS.Calo.hgcalTestGuardRingEE_cfi import *

hgcalTestGuardRingHE = hgcalTestGuardRingEE.clone(
    nameSense = "HGCalHESiliconSensitive",
    waferFile = "testWafersHE.txt"
)
