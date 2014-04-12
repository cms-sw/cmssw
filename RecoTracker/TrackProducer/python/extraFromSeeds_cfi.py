import FWCore.ParameterSet.Config as cms

extraFromSeeds = cms.EDProducer("ExtraFromSeeds",
                                tracks = cms.InputTag('generalTracks')
                                )
