import FWCore.ParameterSet.Config as cms

import SimMuon.MCTruth.MuonAssociatorByHits_cfi
tpToStaMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaMuonAssociation.SimToReco_useTracker = False
tpToStaMuonAssociation.SimToReco_useMuon = True
tpToStaMuonAssociation.EfficiencyCut_muon = 0.5
tpToStaMuonAssociation.PurityCut_muon = 0.5

