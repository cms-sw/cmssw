import FWCore.ParameterSet.Config as cms

validationMuonRPCRecHits = cms.EDFilter("RPCRecHitValid",
    # Label to retrieve Digis from the event 
    rootFileName = cms.untracked.string('/tmp/trentad/rpcRecHitQualityPlots.root')
)


