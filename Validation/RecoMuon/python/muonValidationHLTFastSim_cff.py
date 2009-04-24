import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.muonValidationHLT_cff import *

# Configurations for MultiTrackValidators

l2MuonTrackV.associatormap = 'tpToL2TrackAssociationFS'
l2MuonTrackV.beamSpot = 'offlineBeamSpot'

l2UpdMuonTrackV.associatormap = 'tpToL2TrackAssociationFS'
l2UpdMuonTrackV.beamSpot = 'offlineBeamSpot'

l3MuonTrackV.associatormap = 'tpToL3TrackAssociationFS'
l3MuonTrackV.beamSpot = 'offlineBeamSpot'

l3TkMuonTrackV.associatormap = 'tpToL3TkTrackTrackAssociationFS'
l3TkMuonTrackV.beamSpot = 'offlineBeamSpot'

l3TkMuonMuTrackV.associatormap = 'tpToL3TkMuonAssociationFS'
l3TkMuonMuTrackV.beamSpot = 'offlineBeamSpot'

l2MuonMuTrackV.associatormap = 'tpToL2MuonAssociationFS'
l2MuonMuTrackV.beamSpot = 'offlineBeamSpot'

l2UpdMuonMuTrackV.associatormap = 'tpToL2MuonAssociationFS'
l2UpdMuonMuTrackV.beamSpot = 'offlineBeamSpot'

l3MuonMuTrackV.associatormap = 'tpToL3MuonAssociationFS'
l3MuonMuTrackV.beamSpot = 'offlineBeamSpot'

# # Muon validation sequence
muonValidationHLTFastSim_seq = cms.Sequence(
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
recoMuonValidationHLTFastSim_seq = cms.Sequence(
     muonAssociationHLTFastSim_seq
     *muonValidationHLTFastSim_seq
     )
