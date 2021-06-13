import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackingParticlePrunerByGen_cfi import *
from SimTracker.TrackAssociation.digiSimLinkPruner_cfi import *

from Configuration.Eras.Modifier_fastSim_cff import fastSim
from Configuration.Eras.Modifier_bParking_cff import bParking

tpPruningTask = cms.Task()


_bParking_tpPruningTask = cms.Task(prunedTrackingParticles,prunedDigiSimLinks)
bParking.toReplaceWith(tpPruningTask,_bParking_tpPruningTask)

(fastSim & bParking).toModify(tpPruningTask, lambda x: x.remove(prunedDigiSimLinks))
