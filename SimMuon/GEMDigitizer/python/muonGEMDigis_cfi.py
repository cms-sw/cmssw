import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM digis.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    Noise = cms.bool(True),
    digiModelConfig = cms.PSet(
        signalPropagationSpeed = cms.double(0.66),
        cosmics = cms.bool(False),
        timeResolution = cms.double(2.5),
        timeJitter = cms.double(1.0),
        averageShapingTime = cms.double(50.0),
        timeCalibrationOffset = cms.double(0.0),
        averageClusterSize = cms.double(1.5),
        averageEfficiency = cms.double(0.95),
        averageNoiseRate = cms.double(10.0),
        numberOfStripsPerPartition = cms.int32(384),
        bxwidth = cms.double(25.0),
        minBunch = cms.int32(-5), ## in terms of 25 ns
        maxBunch = cms.int32(3)
    ),
    Signal = cms.bool(True),
    InputCollection = cms.string('g4SimHitsMuonGEMHits'),
    # digiModel = cms.string('GEMSimTriv')
    digiModel = cms.string('GEMSimAverage')
)
