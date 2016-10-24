from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi import *
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(trackingParticleNumberOfLayersProducer, simHits=['famosSimHits:TrackerHits'])
