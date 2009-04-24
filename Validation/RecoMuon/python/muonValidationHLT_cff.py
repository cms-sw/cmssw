import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
import Validation.RecoMuon.MultiTrackValidator_cfi

l2MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l2MuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2MuonTrackV.label = ('hltL2Muons',)
l2MuonTrackV.associatormap = 'tpToL2TrackAssociation'
l2MuonTrackV.associators = 'TrackAssociatorByDeltaR'
l2MuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
l2MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
#l2MuonTrackV.nintHit = 35
#l2MuonTrackV.maxHit = 35.0
l2MuonTrackV.maxpT = 1100.0
l2MuonTrackV.ignoremissingtrackcollection=True

l2UpdMuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l2UpdMuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2UpdMuonTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonTrackV.associatormap = 'tpToL2UpdTrackAssociation'
l2UpdMuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l2UpdMuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
l2UpdMuonTrackV.beamSpot = 'hltOfflineBeamSpot'
#l2UpdMuonTrackV.nintHit = 35
#l2UpdMuonTrackV.maxHit = 35.0
l2UpdMuonTrackV.maxpT = 1100.0
l2UpdMuonTrackV.ignoremissingtrackcollection=True

l3MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3MuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3MuonTrackV.associatormap = 'tpToL3TrackAssociation'
l3MuonTrackV.label = ('hltL3Muons',)
l3MuonTrackV.associators = 'TrackAssociatorByDeltaR'
l3MuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
l3MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
#l3MuonTrackV.nintHit = 35
#l3MuonTrackV.maxHit = 35.0
l3MuonTrackV.maxpT = 1100.0
l3MuonTrackV.ignoremissingtrackcollection=True

l3TkMuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()


l3TkMuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3TkMuonTrackV.associatormap = 'tpToL3TkTrackTrackAssociation'
l3TkMuonTrackV.label = ('hltL3TkTracksFromL2',)
l3TkMuonTrackV.associators = 'TrackAssociatorByHits'
l3TkMuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
l3TkMuonTrackV.beamSpot = 'hltOfflineBeamSpot'
#l3TkMuonTrackV.nintHit = 35
#l3TkMuonTrackV.maxHit = 35.0
l3TkMuonTrackV.maxpT = 1100.0
l3TkMuonTrackV.ignoremissingtrackcollection=True

l3TkMuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3TkMuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3TkMuonMuTrackV.associatormap = 'tpToL3TkMuonAssociation'
l3TkMuonMuTrackV.label = ('hltL3TkTracksFromL2:',)
l3TkMuonMuTrackV.associators = 'muonAssociatorByHits'
l3TkMuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
l3TkMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
#l3TkMuonMuTrackV.nintHit = 35
#l3TkMuonMuTrackV.maxHit = 35.0
l3TkMuonMuTrackV.maxpT = 1100.0
l3TkMuonMuTrackV.ignoremissingtrackcollection=True

l2MuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l2MuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2MuonMuTrackV.associatormap = 'tpToL2MuonAssociation'
l2MuonMuTrackV.label = ('hltL2Muons',)
l2MuonMuTrackV.associators = 'muonAssociatorByHits'
l2MuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
l2MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
#l2MuonMuTrackV.nintHit = 35
#l2MuonMuTrackV.maxHit = 35.0
l2MuonMuTrackV.maxpT = 1100.0
l2MuonMuTrackV.ignoremissingtrackcollection=True

l2UpdMuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l2UpdMuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2UpdMuonMuTrackV.associatormap = 'tpToL2UpdMuonAssociation'
l2UpdMuonMuTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonMuTrackV.associators = 'muonAssociatorByHits'
l2UpdMuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
l2UpdMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
#l2UpdMuonMuTrackV.nintHit = 35
#l2UpdMuonMuTrackV.maxHit = 35.0
l2UpdMuonMuTrackV.maxpT = 1100.0
l2UpdMuonMuTrackV.ignoremissingtrackcollection=True

l3MuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3MuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3MuonMuTrackV.associatormap = 'tpToL3MuonAssociation'
l3MuonMuTrackV.label = ('hltL3Muons:',)
l3MuonMuTrackV.associators = 'muonAssociatorByHits'
l3MuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
l3MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
#l3MuonMuTrackV.nintHit = 35
#l3MuonMuTrackV.maxHit = 35.0
l3MuonMuTrackV.maxpT = 1100.0
l3MuonMuTrackV.ignoremissingtrackcollection=True


# # Muon validation sequence
muonValidationHLT_seq = cms.Sequence(
     l2MuonTrackV
     +l2UpdMuonTrackV
     +l3MuonTrackV
     +l3TkMuonTrackV
     +l3TkMuonMuTrackV
     +l2MuonMuTrackV
     +l2UpdMuonMuTrackV
     +l3MuonMuTrackV
## # #   +recoMuonVMuAssoc
## # #   +recoMuonVTrackAssoc
     )


# The muon HLT association and validation sequence
recoMuonValidationHLT_seq = cms.Sequence(
     muonAssociationHLT_seq
     *muonValidationHLT_seq
     )
