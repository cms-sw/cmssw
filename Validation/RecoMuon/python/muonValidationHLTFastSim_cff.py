import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidationHLT_cff import *

# Configurations for MuonTrackValidators

l2MuonTrackFSV = Validation.RecoMuon.muonValidationHLT_cff.l2MuonTrackV.clone()
l2MuonTrackFSV.associatormap = 'tpToL2TrackAssociationFS'
l2MuonTrackFSV.label_tp_effic = 'mix:MergedTrackTruth'
l2MuonTrackFSV.label_tp_fake = 'mix:MergedTrackTruth'
l2MuonTrackFSV.beamSpot = 'offlineBeamSpot'

l2UpdMuonTrackFSV = Validation.RecoMuon.muonValidationHLT_cff.l2UpdMuonTrackV.clone()
l2UpdMuonTrackFSV.associatormap = 'tpToL2UpdTrackAssociationFS'
l2UpdMuonTrackFSV.label_tp_effic = 'mix:MergedTrackTruth'
l2UpdMuonTrackFSV.label_tp_fake = 'mix:MergedTrackTruth'
l2UpdMuonTrackFSV.beamSpot = 'offlineBeamSpot'

l3TkMuonTrackFSV = Validation.RecoMuon.muonValidationHLT_cff.l3TkMuonTrackV.clone()
l3TkMuonTrackFSV.associatormap = 'tpToL3TkTrackTrackAssociationFS'
l3TkMuonTrackFSV.label_tp_effic = 'mix:MergedTrackTruth'
l3TkMuonTrackFSV.label_tp_fake = 'mix:MergedTrackTruth'
l3TkMuonTrackFSV.beamSpot = 'offlineBeamSpot'

l3MuonTrackFSV = Validation.RecoMuon.muonValidationHLT_cff.l3MuonTrackV.clone()
l3MuonTrackFSV.associatormap = 'tpToL3TrackAssociationFS'
l3MuonTrackFSV.label_tp_effic = 'mix:MergedTrackTruth'
l3MuonTrackFSV.label_tp_fake = 'mix:MergedTrackTruth'
l3MuonTrackFSV.beamSpot = 'offlineBeamSpot'

l2MuonMuTrackFSV = Validation.RecoMuon.muonValidationHLT_cff.l2MuonMuTrackV.clone()
l2MuonMuTrackFSV.associatormap = 'tpToL2MuonAssociationFS'
l2MuonMuTrackFSV.label_tp_effic = 'mix:MergedTrackTruth'
l2MuonMuTrackFSV.label_tp_fake = 'mix:MergedTrackTruth'
l2MuonMuTrackFSV.beamSpot = 'offlineBeamSpot'

l2UpdMuonMuTrackFSV = Validation.RecoMuon.muonValidationHLT_cff.l2UpdMuonMuTrackV.clone()
l2UpdMuonMuTrackFSV.associatormap = 'tpToL2UpdMuonAssociationFS'
l2UpdMuonMuTrackFSV.label_tp_effic = 'mix:MergedTrackTruth'
l2UpdMuonMuTrackFSV.label_tp_fake = 'mix:MergedTrackTruth'
l2UpdMuonMuTrackFSV.beamSpot = 'offlineBeamSpot'

l3TkMuonMuTrackFSV = Validation.RecoMuon.muonValidationHLT_cff.l3TkMuonMuTrackV.clone()
l3TkMuonMuTrackFSV.associatormap = 'tpToL3TkMuonAssociationFS'
l3TkMuonMuTrackFSV.label_tp_effic = 'mix:MergedTrackTruth'
l3TkMuonMuTrackFSV.label_tp_fake = 'mix:MergedTrackTruth'
l3TkMuonMuTrackFSV.beamSpot = 'offlineBeamSpot'

l3MuonMuTrackFSV = Validation.RecoMuon.muonValidationHLT_cff.l3MuonMuTrackV.clone()
l3MuonMuTrackFSV.associatormap = 'tpToL3MuonAssociationFS'
l3MuonMuTrackFSV.label_tp_effic = 'mix:MergedTrackTruth'
l3MuonMuTrackFSV.label_tp_fake = 'mix:MergedTrackTruth'
l3MuonMuTrackFSV.beamSpot = 'offlineBeamSpot'

# # Muon validation sequence
muonValidationHLTFastSim_seq = cms.Sequence(
    l2MuonMuTrackFSV+l2UpdMuonMuTrackFSV+l3MuonMuTrackFSV+l3TkMuonMuTrackFSV
    )


# The muon HLT association and validation sequence
recoMuonAssociationHLTFastSim_seq = cms.Sequence(muonAssociationHLTFastSim_seq)
recoMuonValidationHLTFastSim_seq = cms.Sequence(muonValidationHLTFastSim_seq)
