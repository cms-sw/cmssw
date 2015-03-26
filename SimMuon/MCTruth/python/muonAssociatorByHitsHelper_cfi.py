import FWCore.ParameterSet.Config as cms

from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters
muonAssociatorByHitsHelper = cms.EDProducer("MuonToTrackingParticleAssociatorEDProducer",
    muonAssociatorByHitsCommonParameters
)
