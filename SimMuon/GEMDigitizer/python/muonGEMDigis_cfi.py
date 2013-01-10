import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM digis.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    Noise = cms.bool(True),
    digiModelConfig = cms.PSet(
        signalPropagationSpeed = cms.double(0.66),
        printOutDigitizer = cms.bool(False),
        cosmics = cms.bool(False),
        timeResolution = cms.double(2.5),
        timeJitter = cms.double(1.0),
        averageShapingTime = cms.double(50.0),
        averageClusterSize = cms.double(1.5),
        averageEfficiency = cms.double(0.95),
        averageNoiseRate = cms.double(0.0),
        deltatimeAdjacentStrip = cms.double(3.0),
        rate = cms.double(0.0),
        nbxing = cms.int32(9),
        gate = cms.double(25.0)
    ),
    Signal = cms.bool(True),
    InputCollection = cms.string('g4SimHitsMuonGEMHits'),
    # digiModel = cms.string('GEMSimTriv')
    digiModel = cms.string('GEMSimAverage')
)
