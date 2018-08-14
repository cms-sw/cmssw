import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *
from Validation.RecoMuon.histoParameters_cff import *
import Validation.RecoMuon.MuonTrackValidator_cfi

l2MuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l2MuonMuTrackV.associatormap = 'tpToL2MuonAssociation'
l2MuonMuTrackV.label = ('hltL2Muons',)
l2MuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l2MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l2MuonMuTrackV.ignoremissingtrackcollection=True
l2MuonMuTrackV.muonHistoParameters = staMuonHistoParameters

l2UpdMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l2UpdMuonMuTrackV.associatormap = 'tpToL2UpdMuonAssociation'
l2UpdMuonMuTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2UpdMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l2UpdMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l2UpdMuonMuTrackV.ignoremissingtrackcollection=True
l2UpdMuonMuTrackV.muonHistoParameters = staUpdMuonHistoParameters

l3TkMuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3TkMuonMuTrackV.associatormap = 'tpToL3TkMuonAssociation'
l3TkMuonMuTrackV.label = ('hltL3TkTracksFromL2:',)
l3TkMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l3TkMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3TkMuonMuTrackV.ignoremissingtrackcollection=True
l3TkMuonMuTrackV.muonHistoParameters = trkMuonHistoParameters

l3MuonMuTrackV = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
l3MuonMuTrackV.associatormap = 'tpToL3MuonAssociation'
l3MuonMuTrackV.label = ('hltL3Muons:',)
l3MuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l3MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
l3MuonMuTrackV.ignoremissingtrackcollection=True
l3MuonMuTrackV.muonHistoParameters = glbMuonHistoParameters
#
# The full Muon HLT validation sequence
#
muonValidationHLT_seq = cms.Sequence(
    tpToL2MuonAssociation + l2MuonMuTrackV
    +tpToL2UpdMuonAssociation + l2UpdMuonMuTrackV
    +tpToL3TkMuonAssociation + l3TkMuonMuTrackV
    +tpToL3MuonAssociation + l3MuonMuTrackV
    )

recoMuonValidationHLT_seq = cms.Sequence(
    muonValidationHLT_seq
    )
