import FWCore.ParameterSet.Config as cms

from Validation.HcalHits.hcalSimHitsClient_cfi import *

SimHitsPostProcessor = cms.Sequence(hcalsimhitsClient)
