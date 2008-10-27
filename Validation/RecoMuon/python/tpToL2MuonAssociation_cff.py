import FWCore.ParameterSet.Config as cms

import SimMuon.MCTruth.MuonAssociatorByHits_cfi
tpToL2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL2MuonAssociation.tracksTag = 'hltL2Muons:UpdatedAtVtx'

