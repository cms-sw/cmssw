import FWCore.ParameterSet.Config as cms

SeedToTrackProducer = cms.EDProducer('SeedToTrackProducer',
                                         L2seedsCollection = cms.InputTag("ancientMuonSeed")
                                         )