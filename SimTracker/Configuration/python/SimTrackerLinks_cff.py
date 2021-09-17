import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackingParticlePrunerByGen_cfi import *
from SimTracker.TrackAssociation.digiSimLinkPruner_cfi import *

from Configuration.Eras.Modifier_fastSim_cff import fastSim


tpPruningTask = cms.Task(prunedTrackingParticles,prunedDigiSimLinks)

fastSim.toModify(tpPruningTask, lambda x: x.remove(prunedDigiSimLinks))
