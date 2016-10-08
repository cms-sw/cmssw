from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi import *
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(trackingParticleNumberOfLayersProducer, simHits=['famosSimHits:TrackerHits'])
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(trackingParticleNumberOfLayersProducer, simHits = ["g4SimHits:TrackerHitsPixelBarrelLowTof", "g4SimHits:TrackerHitsPixelEndcapLowTof"])
