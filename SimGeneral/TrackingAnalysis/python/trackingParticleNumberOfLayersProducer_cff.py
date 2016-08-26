from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi import *
from Configuration.StandardSequences.Eras import eras
eras.fastSim.toModify(trackingParticleNumberOfLayersProducer, simHits=['famosSimHits:TrackerHits'])
eras.trackingPhase2PU140.toModify(trackingParticleNumberOfLayersProducer, simHits = cms.VInputTag(cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
                                                                                                  cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"))
)
