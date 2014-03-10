import FWCore.ParameterSet.Config as cms

from Validation.MuonGEMHits.MuonGEMHits_cfi import *
from Validation.MuonGEMDigis.MuonGEMDigis_cfi import *
from Validation.MuonGEMRecHits.MuonGEMRecHits_cfi import *

gemSimValid = cms.Sequence(gemHitsValidation*gemDigiValidation*gemRecHitsValidation)
