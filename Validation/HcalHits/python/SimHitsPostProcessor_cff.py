import FWCore.ParameterSet.Config as cms

from Validation.HcalHits.hcalSimHitsClient_cfi import *

hcalSimHitsPostProcessor = cms.Sequence(hcalsimhitsClient)
