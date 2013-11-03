import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM digis.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    signalPropagationSpeed = cms.double(0.66),
    cosmics = cms.bool(False),
    timeResolution = cms.double(5),
    timeJitter = cms.double(1.0),
    averageShapingTime = cms.double(50.0),
    timeCalibrationOffset = cms.double(19.9),
    averageClusterSize = cms.double(1.5),
    averageEfficiency = cms.double(0.98),
    averageNoiseRate = cms.double(0.0), #intrinsic noise
    numberOfStripsPerPartition = cms.int32(384),
    bxwidth = cms.int32(25),
    minBunch = cms.int32(-5), ## in terms of 25 ns
    maxBunch = cms.int32(3),
    inputCollection = cms.string('g4SimHitsMuonGEMHits'),
    digiModelString = cms.string('Simple'),
    digitizeOnlyMuons = cms.bool(False),
#    neutronGammaRoll = cms.vdouble(76., 62., 52., 45., 39., 30., 23., 18, 0., 0., 0.)#, #n and gamma bkg per roll
    neutronGammaRoll = cms.vdouble(314.1, 199.1, 143.8, 145.5, 121.1, 101.7, 74.0, 69.3, 0., 0., 0.)#, #n, gamma and charged prtcls bkg per roll
)
