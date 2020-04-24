import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.NewSelectors_cff import *
from Validation.RecoMuon.NewAssociators_cff import *
from Validation.RecoMuon.histoParameters_cff import *
import Validation.RecoMuon.NewMuonTrackValidator_cfi

NEWl2MuonMuTrackV = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWl2MuonMuTrackV.associatormap = 'NEWtpToL2MuonAssociation'
NEWl2MuonMuTrackV.label = ('hltL2Muons',)
NEWl2MuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l2MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
NEWl2MuonMuTrackV.ignoremissingtrackcollection=True
NEWl2MuonMuTrackV.muonHistoParameters = staMuonHistoParameters

NEWl2UpdMuonMuTrackV = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWl2UpdMuonMuTrackV.associatormap = 'NEWtpToL2UpdMuonAssociation'
NEWl2UpdMuonMuTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
NEWl2UpdMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l2UpdMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
NEWl2UpdMuonMuTrackV.ignoremissingtrackcollection=True
NEWl2UpdMuonMuTrackV.muonHistoParameters = staUpdMuonHistoParameters

NEWl3TkMuonMuTrackV = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWl3TkMuonMuTrackV.associatormap = 'NEWtpToL3TkMuonAssociation'
NEWl3TkMuonMuTrackV.label = ('hltL3TkTracksFromL2:',)
NEWl3TkMuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l3TkMuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
NEWl3TkMuonMuTrackV.ignoremissingtrackcollection=True
NEWl3TkMuonMuTrackV.muonHistoParameters = trkMuonHistoParameters

NEWl3MuonMuTrackV = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
NEWl3MuonMuTrackV.associatormap = 'NEWtpToL3MuonAssociation'
NEWl3MuonMuTrackV.label = ('hltL3Muons:',)
NEWl3MuonMuTrackV.dirName = 'HLT/Muon/MuonTrack/'
#l3MuonMuTrackV.beamSpot = 'hltOfflineBeamSpot'
NEWl3MuonMuTrackV.ignoremissingtrackcollection=True
NEWl3MuonMuTrackV.muonHistoParameters = glbMuonHistoParameters
#
# The full Muon HLT validation sequence
#
NEWmuonValidationHLT_seq = cms.Sequence(
    NEWtpToL2MuonAssociation + NEWl2MuonMuTrackV
    +NEWtpToL2UpdMuonAssociation + NEWl2UpdMuonMuTrackV
    +NEWtpToL3TkMuonAssociation + NEWl3TkMuonMuTrackV
    +NEWtpToL3MuonAssociation + NEWl3MuonMuTrackV
    )

NEWrecoMuonValidationHLT_seq = cms.Sequence(
    NEWmuonValidationHLT_seq
    )
