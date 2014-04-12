import FWCore.ParameterSet.Config as cms

# Muon Associator
from SimMuon.MCTruth.MuonAssociatorByHitsESProducer_cfi import *

# Setting for StandAloneMuons
muonAssociatorByHitsESProducer.tracksTag = cms.InputTag("standAloneMuons")
muonAssociatorByHitsESProducer.UseTracker = cms.bool(False)
muonAssociatorByHitsESProducer.UseMuon = cms.bool(True)

# Track history parameters
muonHistory = cms.PSet(
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mix","MergedTrackTruth"),
    trackAssociator = cms.untracked.string("muonAssociatorByHits"),
    trackProducer = cms.untracked.InputTag("standAloneMuons"),
    enableRecoToSim = cms.untracked.bool(True),
    enableSimToReco = cms.untracked.bool(False)
)

