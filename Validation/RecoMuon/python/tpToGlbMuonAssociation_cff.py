import FWCore.ParameterSet.Config as cms

import SimMuon.MCTruth.MuonAssociatorByHits_cfi
tpToGlbMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbMuonAssociation.tracksTag = 'globalMuons'
tpToGlbMuonAssociation.SimToReco_useTracker = True
tpToGlbMuonAssociation.SimToReco_useMuon = True
tpToGlbMuonAssociation.EfficiencyCut_muon = 0.5
tpToGlbMuonAssociation.PurityCut_muon = 0.5
tpToGlbMuonAssociation.EfficiencyCut_track = 0.5
tpToGlbMuonAssociation.PurityCut_track = 0.75

