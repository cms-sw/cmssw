from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi import *
from Configuration.StandardSequences.Eras import eras
eras.fastSim.toModify(trackingParticleNumberOfLayersProducer, simHits=['famosSimHits:TrackerHits'])
eras.trackingPhase2PU140.toModify(trackingParticleNumberOfLayersProducer, simHits = ["g4SimHits:TrackerHitsPixelBarrelLowTof", "g4SimHits:TrackerHitsPixelEndcapLowTof"])
