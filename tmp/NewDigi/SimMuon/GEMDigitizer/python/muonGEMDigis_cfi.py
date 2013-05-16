import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM digis.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    numberOfStripsPerPartition = cms.int32(384),
    simulateElectrons = cms.bool(False),                             
    inputCollection = cms.string("g4SimHitsMuonGEMHits"),
    ## choose from Triv, Simple and Detailed
    digiModelString = cms.string("Trivial"),
    timingModelConfig = cms.PSet(
        timeResolution = cms.double(5), ## [ns]
        timeJitter = cms.double(1.0), ## [ns]
        averageShapingTime = cms.double(50.0), ## [ns]
        timeCalibrationOffset = cms.double(19.9), ## [ns]
        bxWidth = cms.double(25.0), ## [ns]
        minBunch = cms.int32(-5), ## in terms of 25 ns
        maxBunch = cms.int32(3),
        signalPropagationSpeed = cms.double(0.66), ## relative
        cosmics = cms.bool(False),
    ),
    noiseModelConfig = cms.PSet(
        averageNoiseRate = cms.double(200.0) ## [Hz/cm^2]
    ),
    clusteringModelConfig = cms.PSet(
        averageClusterSize = cms.double(1.9)
    ),
    efficiencyModelConfig = cms.PSet(
        averageEfficiency = cms.double(0.98)
    )
)
