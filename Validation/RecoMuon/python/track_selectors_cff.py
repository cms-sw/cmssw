import FWCore.ParameterSet.Config as cms

# select probe tracks
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
probeTracks = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
#probeTracks.quality = cms.vstring('highPurity')   # previous setting
#probeTracks.quality = cms.vstring('loose')        # default
probeTracks.tip = cms.double(3.5)
probeTracks.lip = cms.double(30.)
probeTracks.ptMin = cms.double(4.0)
probeTracks.minRapidity = cms.double(-2.4)
probeTracks.maxRapidity = cms.double(2.4)
probeTracks_seq = cms.Sequence( probeTracks )

import SimMuon.MCTruth.MuonTrackProducer_cfi
extractGemMuons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone()
extractGemMuons.selectionTags = ('All',)
extractGemMuons.trackType = "gemMuonTrack"
extractGemMuonsTracks_seq = cms.Sequence( extractGemMuons )

extractMe0Muons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone()
extractMe0Muons.selectionTags = cms.vstring('All',)
extractMe0Muons.trackType = "me0MuonTrack"
extractMe0MuonsTracks_seq = cms.Sequence( extractMe0Muons )

#
# Configuration for Seed track extractor
#

import SimMuon.MCTruth.SeedToTrackProducer_cfi
seedsOfSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone()
seedsOfSTAmuons.L2seedsCollection = cms.InputTag("ancientMuonSeed")
seedsOfSTAmuons_seq = cms.Sequence( seedsOfSTAmuons )

seedsOfDisplacedSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone()
seedsOfDisplacedSTAmuons.L2seedsCollection = cms.InputTag("displacedMuonSeeds")
seedsOfDisplacedSTAmuons_seq = cms.Sequence( seedsOfDisplacedSTAmuons )
