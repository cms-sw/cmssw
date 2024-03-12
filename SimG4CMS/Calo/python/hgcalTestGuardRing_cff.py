import FWCore.ParameterSet.Config as cms

from SimG4CMS.Calo.hgcalTestGuardRingEE_cfi import *

hgcalTestGuardRingHE = hgcalTestGuardRingEE.clone(
    nameSense = "HGCalHESiliconSensitive",
    waferFile = "testWafersHE.txt"
)
# foo bar baz
# zKVcu9Gah2zHj
# jtn1AZJ57PD4A
