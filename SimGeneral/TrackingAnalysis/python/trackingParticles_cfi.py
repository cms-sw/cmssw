import FWCore.ParameterSet.Config as cms

trackingtruthprod = cms.EDProducer("TrackingTruthProducer",
    discardOutVolume = cms.bool(False),
    DiscardHitsFromDeltas = cms.bool(True),
    simHitLabel = cms.string('g4SimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    HepMCDataLabels = cms.vstring('VtxSmeared', 
        'PythiaSource', 
        'source'),
    TrackerHitLabels = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
        'g4SimHitsTrackerHitsPixelBarrelHighTof', 
        'g4SimHitsTrackerHitsPixelEndcapLowTof', 
        'g4SimHitsTrackerHitsPixelEndcapHighTof', 
        'g4SimHitsTrackerHitsTIBLowTof', 
        'g4SimHitsTrackerHitsTIBHighTof', 
        'g4SimHitsTrackerHitsTIDLowTof', 
        'g4SimHitsTrackerHitsTIDHighTof', 
        'g4SimHitsTrackerHitsTOBLowTof', 
        'g4SimHitsTrackerHitsTOBHighTof', 
        'g4SimHitsTrackerHitsTECLowTof', 
        'g4SimHitsTrackerHitsTECHighTof'),
    volumeZ = cms.double(3000.0)
)

electrontruth = cms.EDProducer("TrackingElectronProducer")

mergedtruth = cms.EDProducer("MergedTruthProducer")

trackingParticles = cms.Sequence(trackingtruthprod*electrontruth*mergedtruth)


