import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MuonTrackValidators
import Validation.RecoMuon.MuonTrackValidator_cfi

l2MuonTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l2MuonTrackV.label_tp_effic = 'mix:MergedTrackTruth'
l2MuonTrackV.label = ('hltL2Muons',)
l2MuonTrackV.associatormap = 'tpToL2TrackAssociation'
l2MuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l2MuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l2MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l2MuonTrackV.ignoremissingtrackcollection=True
l2MuonTrackV.usetracker = False
l2MuonTrackV.usemuon = True

l2UpdMuonTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l2UpdMuonTrackV.label_tp_effic = 'mix:MergedTrackTruth'
l2UpdMuonTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonTrackV.associatormap = 'tpToL2UpdTrackAssociation'
l2UpdMuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l2UpdMuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l2UpdMuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l2UpdMuonTrackV.ignoremissingtrackcollection=True
l2UpdMuonTrackV.usetracker = False
l2UpdMuonTrackV.usemuon = True

l3MuonTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3MuonTrackV.label_tp_effic = 'mix:MergedTrackTruth'
l3MuonTrackV.associatormap = 'tpToL3TrackAssociation'
l3MuonTrackV.label = ('hltL3Muons',)
l3MuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l3MuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l3MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonTrackV.ignoremissingtrackcollection=True
l3MuonTrackV.usetracker = True
l3MuonTrackV.usemuon = True

l3TkMuonTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3TkMuonTrackV.label_tp_effic = 'mix:MergedTrackTruth'
l3TkMuonTrackV.associatormap = 'tpToL3TkTrackTrackAssociation'
l3TkMuonTrackV.label = ('hltL3TkTracksFromL2',)
l3TkMuonTrackV.associators = ('OnlineTrackAssociatorByHits',)
l3TkMuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l3TkMuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l3TkMuonTrackV.ignoremissingtrackcollection=True
l3TkMuonTrackV.usetracker = True
l3TkMuonTrackV.usemuon = False

l3TkMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3TkMuonMuTrackV.label_tp_effic = 'mix:MergedTrackTruth'
l3TkMuonMuTrackV.associatormap = 'tpToL3TkMuonAssociation'
l3TkMuonMuTrackV.label = ('hltL3TkTracksFromL2:',)
l3TkMuonMuTrackV.associators = ('MuonAssociationByHits',)
l3TkMuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l3TkMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3TkMuonMuTrackV.ignoremissingtrackcollection=True
l3TkMuonMuTrackV.usetracker = True
l3TkMuonMuTrackV.usemuon = False

l2MuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l2MuonMuTrackV.label_tp_effic = 'mix:MergedTrackTruth'
l2MuonMuTrackV.associatormap = 'tpToL2MuonAssociation'
l2MuonMuTrackV.label = ('hltL2Muons',)
l2MuonMuTrackV.associators = ('MuonAssociationByHits',)
l2MuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l2MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l2MuonMuTrackV.ignoremissingtrackcollection=True
l2MuonMuTrackV.usetracker = False
l2MuonMuTrackV.usemuon = True

l2UpdMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l2UpdMuonMuTrackV.label_tp_effic = 'mix:MergedTrackTruth'
l2UpdMuonMuTrackV.associatormap = 'tpToL2UpdMuonAssociation'
l2UpdMuonMuTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonMuTrackV.associators = ('MuonAssociationByHits',)
l2UpdMuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l2UpdMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l2UpdMuonMuTrackV.ignoremissingtrackcollection=True
l2UpdMuonMuTrackV.usetracker = False
l2UpdMuonMuTrackV.usemuon = True

l3MuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3MuonMuTrackV.label_tp_effic = 'mix:MergedTrackTruth'
l3MuonMuTrackV.associatormap = 'tpToL3MuonAssociation'
l3MuonMuTrackV.label = ('hltL3Muons:',)
l3MuonMuTrackV.associators = ('MuonAssociationByHits',)
l3MuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
#l3MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonMuTrackV.ignoremissingtrackcollection=True
l3MuonMuTrackV.usetracker = True
l3MuonMuTrackV.usemuon = True


# # Muon HLT validation sequence
muonValidationHLT_seq = cms.Sequence(
    l2MuonMuTrackV+l2UpdMuonMuTrackV+l3TkMuonMuTrackV+l3MuonMuTrackV
    )


# The muon HLT association and validation sequence
recoMuonValidationHLT_seq = cms.Sequence(
    muonAssociationHLT_seq
    *muonValidationHLT_seq
    )
