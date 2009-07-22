import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
import Validation.RecoMuon.MultiTrackValidator_cfi

l2MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()

l2MuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2MuonTrackV.label = ('hltL2Muons',)
l2MuonTrackV.associatormap = 'tpToL2TrackAssociation'
l2MuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l2MuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l2MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l2MuonTrackV.ignoremissingtrackcollection=True

l2UpdMuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()

l2UpdMuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2UpdMuonTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonTrackV.associatormap = 'tpToL2UpdTrackAssociation'
l2UpdMuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l2UpdMuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l2UpdMuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l2UpdMuonTrackV.ignoremissingtrackcollection=True

l3MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()

l3MuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3MuonTrackV.associatormap = 'tpToL3TrackAssociation'
l3MuonTrackV.label = ('hltL3Muons',)
l3MuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l3MuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l3MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonTrackV.ignoremissingtrackcollection=True

l3TkMuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()


l3TkMuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3TkMuonTrackV.associatormap = 'tpToL3TkTrackTrackAssociation'
l3TkMuonTrackV.label = ('hltL3TkTracksFromL2',)
l3TkMuonTrackV.associators = ('OnlineTrackAssociatorByHits',)
l3TkMuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l3TkMuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l3TkMuonTrackV.ignoremissingtrackcollection=True

l3TkMuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()

l3TkMuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3TkMuonMuTrackV.associatormap = 'tpToL3TkMuonAssociation'
l3TkMuonMuTrackV.label = ('hltL3TkTracksFromL2:',)
l3TkMuonMuTrackV.associators = ('tpToL3TkMuonAssociation',)
l3TkMuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l3TkMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3TkMuonMuTrackV.ignoremissingtrackcollection=True

l2MuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()

l2MuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2MuonMuTrackV.associatormap = 'tpToL2MuonAssociation'
l2MuonMuTrackV.label = ('hltL2Muons',)
l2MuonMuTrackV.associators = ('tpToL2MuonAssociation',)
l2MuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l2MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l2MuonMuTrackV.ignoremissingtrackcollection=True

l2UpdMuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()

l2UpdMuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2UpdMuonMuTrackV.associatormap = 'tpToL2UpdMuonAssociation'
l2UpdMuonMuTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonMuTrackV.associators = ('tpToL2UpdMuonAssociation',)
l2UpdMuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l2UpdMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l2UpdMuonMuTrackV.ignoremissingtrackcollection=True

l3MuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.RMmultiTrackValidator.clone()

l3MuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3MuonMuTrackV.associatormap = 'tpToL3MuonAssociation'
l3MuonMuTrackV.label = ('hltL3Muons:',)
l3MuonMuTrackV.associators = ('tpToL3MuonAssociation',)
l3MuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l3MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonMuTrackV.ignoremissingtrackcollection=True


# # Muon HLT validation sequence
muonValidationHLT_seq = cms.Sequence(
     l2MuonTrackV
     +l2UpdMuonTrackV
     +l3MuonTrackV
     +l3TkMuonTrackV
     +l3TkMuonMuTrackV
     +l2MuonMuTrackV
     +l2UpdMuonMuTrackV
     +l3MuonMuTrackV
     )


# The muon HLT association and validation sequence
recoMuonValidationHLT_seq = cms.Sequence(
     muonAssociationHLT_seq
     *muonValidationHLT_seq
     )
