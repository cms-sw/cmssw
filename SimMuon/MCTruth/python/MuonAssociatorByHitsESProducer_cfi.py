import FWCore.ParameterSet.Config as cms

from SimMuon.MCTruth.MuonAssociatorByHits_cfi import muonAssociatorByHitsCommonParameters
muonAssociatorByHitsESProducer = cms.ESProducer("MuonAssociatorESProducer",
    muonAssociatorByHitsCommonParameters,
    ComponentName = cms.string("muonAssociatorByHits"),
)
