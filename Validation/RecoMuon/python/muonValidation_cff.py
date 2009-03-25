import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
from Validation.RecoMuon.muonValidationBase_cff import *

# The muon association and validation sequence
recoMuonValidation = cms.Sequence(muonAssociation_seq*muonValidation_seq)
