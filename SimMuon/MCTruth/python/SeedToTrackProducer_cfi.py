import FWCore.ParameterSet.Config as cms

SeedToTrackProducer = cms.EDProducer('SeedToTrackProducer',
                                         L2seedsCollection = cms.InputTag("ancientMuonSeed")
                                         )


Phase2SeedToTrackProducer = cms.EDProducer('Phase2SeedToTrackProducer',
                                         L2seedsCollection = cms.InputTag("hltL2MuonSeedsFromL1TkMuon")
                                         )

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(SeedToTrackProducer, Phase2SeedToTrackProducer)