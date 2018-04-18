from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi import *
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(trackingParticleNumberOfLayersProducer, simHits=['fastSimProducer:TrackerHits'])
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(trackingParticleNumberOfLayersProducer, simHits = ["g4SimHits:TrackerHitsPixelBarrelLowTof", "g4SimHits:TrackerHitsPixelEndcapLowTof"])

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(trackingParticleNumberOfLayersProducer, trackingParticles = "mixData:MergedTrackTruth")
