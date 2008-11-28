import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
import Validation.RecoMuon.MultiTrackValidator_cfi

l2MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l2MuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2MuonTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2MuonTrackV.associatormap = 'tpToL2TrackAssociation'
l2MuonTrackV.associators = 'TrackAssociatorByDeltaR'
l2MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l2MuonTrackV.nintHit = 35
l2MuonTrackV.maxHit = 35.0
l2MuonTrackV.maxpT = 1100.0

l3MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3MuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3MuonTrackV.associatormap = 'tpToL3TrackAssociation'
l3MuonTrackV.label = ('hltL3Muons',)
l3MuonTrackV.associators = 'TrackAssociatorByDeltaR'
l3MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonTrackV.nintHit = 35
l3MuonTrackV.maxHit = 35.0
l3MuonTrackV.maxpT = 1100.0

# Muon validation sequence
muonValidationHLT_seq = cms.Sequence(l2MuonTrackV+l3MuonTrackV)

recoMuonValidationHLT = cms.Sequence(muonSelector_seq*muonAssociation_seq*muonValidationHLT_seq)
