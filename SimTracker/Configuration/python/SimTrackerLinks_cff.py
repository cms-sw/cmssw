import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackingParticlePrunerByGen_cfi import *
from SimTracker.TrackAssociation.digiSimLinkPruner_cfi import *


tpPruningTask = cms.Task(prunedTrackingParticles,prunedDigiSimLinks)
