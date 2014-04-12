import FWCore.ParameterSet.Config as cms

from Validation.HcalDigis.HcalDigisClient_cfi import *

hcaldigisPostProcessor = cms.Sequence(hcaldigisClient)
