import FWCore.ParameterSet.Config as cms

mergedtruth = cms.EDProducer("TrackingTruthProducer",
    simHitLabel = cms.string('g4SimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    mergedBremsstrahlung = cms.bool(True),
    HepMCDataLabels = cms.vstring('VtxSmeared', 
        'generator', 
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
        'g4SimHitsTrackerHitsTECHighTof',
      	'g4SimHitsMuonDTHits',
	'g4SimHitsMuonCSCHits',
	'g4SimHitsMuonRPCHits'
    ),        
    volumeZ = cms.double(3000.0),
    select = cms.PSet(
        lipTP = cms.double(1000),
        chargedOnlyTP = cms.bool(True),
        pdgIdTP = cms.vint32(),
        signalOnlyTP = cms.bool(True),
        minRapidityTP = cms.double(-2.6),
        minHitTP = cms.int32(3),
        ptMinTP = cms.double(0.2),
        maxRapidityTP = cms.double(2.6),
        tipTP = cms.double(1000)
    )
)

trackingParticleSelection = cms.Sequence(mergedtruth)

# Uncomment in case of running 3 producer approach

# electrontruth = cms.EDProducer("TrackingElectronProducer")

# mergedtruth = cms.EDProducer("MergedTruthProducer")

# trackingParticles = cms.Sequence(trackingtruthprod*electrontruth*mergedtruth)

