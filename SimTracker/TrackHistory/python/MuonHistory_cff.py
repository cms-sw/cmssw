import FWCore.ParameterSet.Config as cms

# Muon Associator
from SimMuon.MCTruth.muonAssociatorByHitsHelper_cfi import *

# Setting for StandAloneMuons
muonAssociatorByHitsHelper.tracksTag = cms.InputTag("standAloneMuons")
muonAssociatorByHitsHelper.UseTracker = cms.bool(False)
muonAssociatorByHitsHelper.UseMuon = cms.bool(True)

# Track history parameters
muonHistory = cms.PSet(
    bestMatchByMaxValue = cms.untracked.bool(True),
    trackingTruth = cms.untracked.InputTag("mix","MergedTrackTruth"),
    trackAssociator = cms.untracked.string("muonAssociatorByHitsHelper"),
    trackProducer = cms.untracked.InputTag("standAloneMuons"),
    enableRecoToSim = cms.untracked.bool(True),
    enableSimToReco = cms.untracked.bool(False)
)

