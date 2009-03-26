import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

# Configurations for MultiTrackValidators
import Validation.RecoMuon.MultiTrackValidator_cfi

l2MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()


l2MuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2MuonTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2MuonTrackV.associatormap = 'tpToL2TrackAssociationFS'
l2MuonTrackV.associators = 'TrackAssociatorByDeltaR'
l2MuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
l2MuonTrackV.beamSpot = 'offlineBeamSpot'
l2MuonTrackV.nintHit = 35
l2MuonTrackV.maxHit = 35.0
l2MuonTrackV.maxpT = 1100.0
l2MuonTrackV.ignoremissingtrackcollection=True

l3MuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3MuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3MuonTrackV.associatormap = 'tpToL3TrackAssociationFS'
l3MuonTrackV.label = ('hltL3Muons',)
l3MuonTrackV.associators = 'TrackAssociatorByDeltaR'
l3MuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
l3MuonTrackV.beamSpot = 'offlineBeamSpot'
l3MuonTrackV.nintHit = 35
l3MuonTrackV.maxHit = 35.0
l3MuonTrackV.maxpT = 1100.0
l3MuonTrackV.ignoremissingtrackcollection=True

l3TkMuonTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()


l3TkMuonTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3TkMuonTrackV.associatormap = 'tpToL3TkTrackTrackAssociationFS'
l3TkMuonTrackV.label = ('hltL3TkTracksFromL2',)
l3TkMuonTrackV.associators = 'TrackAssociatorByHits'
l3TkMuonTrackV.dirName = 'HLT/Muon/MultiTrack/'
l3TkMuonTrackV.beamSpot = 'offlineBeamSpot'
l3TkMuonTrackV.nintHit = 35
l3TkMuonTrackV.maxHit = 35.0
l3TkMuonTrackV.maxpT = 1100.0
l3TkMuonTrackV.ignoremissingtrackcollection=True

l3TkMuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3TkMuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3TkMuonMuTrackV.associatormap = 'tpToL3TkMuonAssociationFS'
l3TkMuonMuTrackV.label = ('hltL3TkTracksFromL2:',)
l3TkMuonMuTrackV.associators = 'muonAssociatorByHits'
l3TkMuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
l3TkMuonMuTrackV.beamSpot = 'offlineBeamSpot'
l3TkMuonMuTrackV.nintHit = 35
l3TkMuonMuTrackV.maxHit = 35.0
l3TkMuonMuTrackV.maxpT = 1100.0
l3TkMuonMuTrackV.ignoremissingtrackcollection=True

l2MuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l2MuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l2MuonMuTrackV.associatormap = 'tpToL2MuonAssociationFS'
l2MuonMuTrackV.label = ('hltL2Muons:UpdatedAtVtx',)
l2MuonMuTrackV.associators = 'muonAssociatorByHits'
l2MuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
l2MuonMuTrackV.beamSpot = 'offlineBeamSpot'
l2MuonMuTrackV.nintHit = 35
l2MuonMuTrackV.maxHit = 35.0
l2MuonMuTrackV.maxpT = 1100.0
l2MuonMuTrackV.ignoremissingtrackcollection=True

l3MuonMuTrackV = Validation.RecoMuon.MultiTrackValidator_cfi.multiTrackValidator.clone()

l3MuonMuTrackV.label_tp_effic = 'mergedtruth:MergedTrackTruth'
l3MuonMuTrackV.associatormap = 'tpToL3MuonAssociationFS'
l3MuonMuTrackV.label = ('hltL3Muons:',)
l3MuonMuTrackV.associators = 'muonAssociatorByHits'
l3MuonMuTrackV.dirName = 'HLT/Muon/MultiTrack/'
l3MuonMuTrackV.beamSpot = 'offlineBeamSpot'
l3MuonMuTrackV.nintHit = 35
l3MuonMuTrackV.maxHit = 35.0
l3MuonMuTrackV.maxpT = 1100.0
l3MuonMuTrackV.ignoremissingtrackcollection=True


# # Muon validation sequence
muonValidationHLTFastSim_seq = cms.Sequence(
     l2MuonTrackV
     +l3MuonTrackV
     +l3TkMuonTrackV
     +l3TkMuonMuTrackV
     +l2MuonMuTrackV
     +l3MuonMuTrackV
## # #   +recoMuonVMuAssoc
## # #   +recoMuonVTrackAssoc
     )


# The muon HLT association and validation sequence
recoMuonValidationHLTFastSim_seq = cms.Sequence(
     muonAssociationHLTFastSim_seq
     *muonValidationHLTFastSim_seq
     )
