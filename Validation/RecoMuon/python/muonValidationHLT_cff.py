import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
from Validation.RecoMuon.muonValidationHLTBase_cff import *

# The muon HLT association and validation sequence
recoMuonValidationHLT_seq = cms.Sequence(
     muonAssociationHLT_seq
     *muonValidationHLT_seq
     )
