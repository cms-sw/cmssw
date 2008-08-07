import FWCore.ParameterSet.Config as cms

import SimMuon.MCTruth.MuonAssociatorByHits_cfi
tpToL3MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3MuonAssociation.tracksTag = 'hltL3Muons'

