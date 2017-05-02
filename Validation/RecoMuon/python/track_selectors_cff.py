import FWCore.ParameterSet.Config as cms

# select probe tracks
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
NEWprobeTracks = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
#NEWprobeTracks.quality = cms.vstring('highPurity')   # previous setting
#NEWprobeTracks.quality = cms.vstring('loose')        # default
NEWprobeTracks.tip = cms.double(3.5)
NEWprobeTracks.lip = cms.double(30.)
NEWprobeTracks.ptMin = cms.double(4.0)
NEWprobeTracks.minRapidity = cms.double(-2.4)
NEWprobeTracks.maxRapidity = cms.double(2.4)
NEWprobeTracks_seq = cms.Sequence( NEWprobeTracks )

import SimMuon.MCTruth.MuonTrackProducer_cfi
NEWextractGemMuons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone()
NEWextractGemMuons.selectionTags = ('All',)
NEWextractGemMuons.trackType = "gemMuonTrack"
NEWextractGemMuonsTracks_seq = cms.Sequence( NEWextractGemMuons )

NEWextractMe0Muons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone()
NEWextractMe0Muons.selectionTags = cms.vstring('All',)
NEWextractMe0Muons.trackType = "me0MuonTrack"
NEWextractMe0MuonsTracks_seq = cms.Sequence( NEWextractMe0Muons )

#
# Configuration for Seed track extractor
#

import SimMuon.MCTruth.SeedToTrackProducer_cfi
NEWseedsOfSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone()
NEWseedsOfSTAmuons.L2seedsCollection = cms.InputTag("ancientMuonSeed")
NEWseedsOfSTAmuons_seq = cms.Sequence( NEWseedsOfSTAmuons )

NEWseedsOfDisplacedSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone()
NEWseedsOfDisplacedSTAmuons.L2seedsCollection = cms.InputTag("displacedMuonSeeds")
NEWseedsOfDisplacedSTAmuons_seq = cms.Sequence( NEWseedsOfDisplacedSTAmuons )
