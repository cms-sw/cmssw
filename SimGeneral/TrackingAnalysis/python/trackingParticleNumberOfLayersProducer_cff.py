from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi import *
from Configuration.StandardSequences.Eras import eras
eras.fastSim.toModify(trackingParticleNumberOfLayersProducer, simHits=['famosSimHits:TrackerHits'])
