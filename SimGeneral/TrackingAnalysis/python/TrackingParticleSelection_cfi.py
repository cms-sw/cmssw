import FWCore.ParameterSet.Config as cms

mergedtruth = cms.EDProducer("TrackingTruthProducer",
 
    mixLabel = cms.string('mix'),
    simHitLabel = cms.string('g4SimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    volumeZ = cms.double(3000.0),
    mergedBremsstrahlung = cms.bool(True),
    removeDeadModules = cms.bool(False),
 
    HepMCDataLabels = cms.vstring('generatorSmeared', 
        'generator', 
        'PythiaSource', 
        'source'
    ),

    useMultipleHepMCLabels = cms.bool(False), 
 
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
    ),

    select = cms.PSet(
        lipTP = cms.double(1000),
        chargedOnlyTP = cms.bool(True),
        pdgIdTP = cms.vint32(),
        signalOnlyTP = cms.bool(True),
        intimeOnlyTP = cms.bool(False),
        minRapidityTP = cms.double(-2.6),
        minHitTP = cms.int32(3),
        ptMinTP = cms.double(0.2),
        maxRapidityTP = cms.double(2.6),
        tipTP = cms.double(1000)
    )
)

trackingParticleSelection = cms.Sequence(mergedtruth)
