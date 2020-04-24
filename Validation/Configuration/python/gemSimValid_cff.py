import FWCore.ParameterSet.Config as cms

from Validation.MuonGEMHits.MuonGEMHits_cff import *
from Validation.MuonGEMDigis.MuonGEMDigis_cff import *
from Validation.MuonGEMRecHits.MuonGEMRecHits_cff import *

gemSimValid = cms.Sequence(gemSimValidation*gemDigiValidation*gemLocalRecoValidation)
