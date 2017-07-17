# configuration for FullSim: muon track validation using TrackAssociatorByPosition
#  (backup solution, incomplete, not run by default)
#
import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.tabp_associators_cff import *
import Validation.RecoMuon.MuonTrackValidator_cfi

l2MuonTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l2MuonTrackV.label = ('hltL2Muons',)
l2MuonTrackV.associatormap = 'tpToL2TrackAssociation'
l2MuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l2MuonTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l2MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l2MuonTrackV.ignoremissingtrackcollection=True
l2MuonTrackV.usetracker = False
l2MuonTrackV.usemuon = True

l2UpdMuonTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l2UpdMuonTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonTrackV.associatormap = 'tpToL2UpdTrackAssociation'
l2UpdMuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l2UpdMuonTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l2UpdMuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l2UpdMuonTrackV.ignoremissingtrackcollection=True
l2UpdMuonTrackV.usetracker = False
l2UpdMuonTrackV.usemuon = True

l3MuonTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3MuonTrackV.associatormap = 'tpToL3TrackAssociation'
l3MuonTrackV.label = ('hltL3Muons',)
l3MuonTrackV.associators = ('TrackAssociatorByDeltaR',)
l3MuonTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l3MuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonTrackV.ignoremissingtrackcollection=True
l3MuonTrackV.usetracker = True
l3MuonTrackV.usemuon = True

l3TkMuonTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3TkMuonTrackV.associatormap = 'tpToL3TkTrackTrackAssociation'
l3TkMuonTrackV.label = ('hltL3TkTracksFromL2',)
l3TkMuonTrackV.associators = ('OnlineTrackAssociatorByHits',)
l3TkMuonTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l3TkMuonTrackV.beamSpot = 'hltOfflineBeamSpot'
l3TkMuonTrackV.ignoremissingtrackcollection=True
l3TkMuonTrackV.usetracker = True
l3TkMuonTrackV.usemuon = False

#
# The full Muon HLT validation sequence
#
muonValidationHLT_seq = cms.Sequence(
    tpToL2TrackAssociation + l2MuonTrackV
    +tpToL2UpdTrackAssociation + l2UpdMuonTrackV
    +tpToL3TkTrackTrackAssociation + l3TkMuonTrackV
    +tpToL3TrackAssociation + l3MuonTrackV
    )

recoMuonValidationHLT_seq = cms.Sequence(muonValidationHLT_seq)
