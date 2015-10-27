import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

from SimTracker.TrackHistory.MuonHistory_cff import *
from SimTracker.TrackHistory.TrackQuality_cff import *

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

MuonClassifier = cms.PSet(
    muonHistory,
    trackQuality,
    hepMC = cms.untracked.InputTag("generatorSmeared"),
    beamSpot = cms.untracked.InputTag("offlineBeamSpot"),
    badPull = cms.untracked.double(3.0),
    longLivedDecayLength = cms.untracked.double(1e-14),
    vertexClusteringDistance = cms.untracked.double(0.003),
    numberOfInnerLayers = cms.untracked.uint32(2),
    minTrackerSimHits = cms.untracked.uint32(3)
)
