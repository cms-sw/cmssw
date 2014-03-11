import FWCore.ParameterSet.Config as cms

from Validation.MuonGEMHits.PostProcessor_cff import *
from Validation.MuonGEMDigis.PostProcessor_cff import *

gemPostValidation = cms.Sequence(MuonGEMHitsPostProcessors*MuonGEMDigisPostProcessors)
