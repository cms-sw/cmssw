import FWCore.ParameterSet.Config as cms

mergedtruth = cms.EDProducer("TrackingTruthProducer",
    simHitLabel = cms.string('g4SimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    volumeZ = cms.double(3000.0),
    mergedBremsstrahlung = cms.bool(True),
    removeDeadModules = cms.bool(False),
    HepMCDataLabels = cms.vstring('VtxSmeared', 
        'generator', 
        'PythiaSource', 
        'source'),

    simHitCollections = cms.PSet(
        pixel = cms.vstring (
            'g4SimHitsTrackerHitsPixelBarrelLowTof',
            'g4SimHitsTrackerHitsPixelBarrelHighTof',
            'g4SimHitsTrackerHitsPixelEndcapLowTof',
            'g4SimHitsTrackerHitsPixelEndcapHighTof'
        ),
        tracker = cms.vstring (
            'g4SimHitsTrackerHitsTIBLowTof',
            'g4SimHitsTrackerHitsTIBHighTof',
            'g4SimHitsTrackerHitsTIDLowTof',
            'g4SimHitsTrackerHitsTIDHighTof',
            'g4SimHitsTrackerHitsTOBLowTof',
            'g4SimHitsTrackerHitsTOBHighTof',
            'g4SimHitsTrackerHitsTECLowTof',
            'g4SimHitsTrackerHitsTECHighTof'
        ),
        muon = cms.vstring (
            'g4SimHitsMuonDTHits',
            'g4SimHitsMuonCSCHits',
            'g4SimHitsMuonRPCHits'            
        )
    )
)

trackingParticles = cms.Sequence(mergedtruth)

# Uncomment in case of running 3 producer approach

# electrontruth = cms.EDProducer("TrackingElectronProducer")

# mergedtruth = cms.EDProducer("MergedTruthProducer")

# trackingParticles = cms.Sequence(trackingtruthprod*electrontruth*mergedtruth)

