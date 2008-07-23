import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

from SimTracker.TrackHistory.TrackHistory_cff import *

trackClassifier = cms.PSet(
    trackHistory,
    badD0Pull = cms.untracked.double(3.0),
    longLivedDecayLenght = cms.untracked.double(1e-14),
    vertexClusteringDistance = cms.untracked.double(0.0001)
)

