import FWCore.ParameterSet.Config as cms

# select probe tracks
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
probeTracks = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone(
#quality = 'highPurity'   # previous setting
#quality = ['loose']        # default
    tip = 3.5,
    lip = 30.,
    ptMin = 3.0,
    minRapidity = -2.4,
    maxRapidity = 2.4
)
probeTracks_seq = cms.Sequence( probeTracks )

# tracks extracted from reco::Muons
import SimMuon.MCTruth.MuonTrackProducer_cfi

extractGemMuons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone(
    selectionTags = ['All'],
    trackType = "gemMuonTrack"
)
extractGemMuonsTracks_seq = cms.Sequence( extractGemMuons )

extractMe0Muons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone(
    selectionTags = ['All'],
    trackType = "me0MuonTrack"
)
extractMe0MuonsTracks_seq = cms.Sequence( extractMe0Muons )

tunepMuonTracks = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone(
    muonsTag = "muons",
    selectionTags = ['All'],
    trackType = "tunepTrack"
)
tunepMuonTracks_seq = cms.Sequence( tunepMuonTracks )

pfMuonTracks = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone(
    muonsTag = "muons",
    selectionTags = ['All'],
    trackType = "pfTrack"
)
pfMuonTracks_seq = cms.Sequence( pfMuonTracks )

recoMuonTracks = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone(
    muonsTag = "muons",
    selectionTags = ['All'],
    trackType = "recomuonTrack"
)
recoMuonTracks_seq = cms.Sequence( recoMuonTracks )

hltIterL3MuonsNoIDTracks = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone(
    muonsTag = "hltIterL3MuonsNoID",
    selectionTags = ['All'],
    trackType = "recomuonTrack",
    ignoreMissingMuonCollection = True
)
hltIterL3MuonsNoIDTracks_seq = cms.Sequence( hltIterL3MuonsNoIDTracks )

hltIterL3MuonsTracks = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone(
    muonsTag = "hltIterL3Muons",
    selectionTags = ['All'],
    trackType = "recomuonTrack",
    ignoreMissingMuonCollection = True
)
hltIterL3MuonsTracks_seq = cms.Sequence( hltIterL3MuonsTracks )

#
# Configuration for Seed track extractor
#

import SimMuon.MCTruth.SeedToTrackProducer_cfi
seedsOfSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone(
    L2seedsCollection = "ancientMuonSeed"
)
seedsOfSTAmuons_seq = cms.Sequence( seedsOfSTAmuons )
seedsOfDisplacedSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone(
    L2seedsCollection = "displacedMuonSeeds"
)
seedsOfDisplacedSTAmuons_seq = cms.Sequence( seedsOfDisplacedSTAmuons )
