import FWCore.ParameterSet.Config as cms

trackingtruthprod = cms.EDProducer("TrackingTruthProducer",
    discardOutVolume = cms.bool(False),
    DiscardHitsFromDeltas = cms.bool(True),
    simHitLabel = cms.string('g4SimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    HepMCDataLabels = cms.vstring('VtxSmeared', 'PythiaSource', 'source'),
    TrackerHitLabels = cms.vstring('TrackerHitsPixelBarrelLowTof', 'TrackerHitsPixelBarrelHighTof', 'TrackerHitsPixelEndcapLowTof', 'TrackerHitsPixelEndcapHighTof', 'TrackerHitsTIBLowTof', 'TrackerHitsTIBHighTof', 'TrackerHitsTIDLowTof', 'TrackerHitsTIDHighTof', 'TrackerHitsTOBLowTof', 'TrackerHitsTOBHighTof', 'TrackerHitsTECLowTof', 'TrackerHitsTECHighTof'),
    volumeZ = cms.double(3000.0)
)

electrontruth = cms.EDProducer("TrackingElectronProducer")

mergedtruth = cms.EDProducer("MergedTruthProducer")

trackingParticles = cms.Sequence(trackingtruthprod*electrontruth*mergedtruth)

