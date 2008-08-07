import FWCore.ParameterSet.Config as cms

import SimMuon.MCTruth.MuonAssociatorByHits_cfi
tpToTkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTkMuonAssociation.tracksTag = 'generalTracks'
tpToTkMuonAssociation.SimToReco_useTracker = True
tpToTkMuonAssociation.SimToReco_useMuon = False
tpToTkMuonAssociation.EfficiencyCut_track = 0.5
tpToTkMuonAssociation.PurityCut_track = 0.75

