import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM digis.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    signalPropagationSpeed = cms.double(0.66),
    cosmics = cms.bool(False),
    timeResolution = cms.double(5),
    timeJitter = cms.double(1.0),
    averageShapingTime = cms.double(50.0),
    timeCalibrationOffset1 = cms.double(19.9),
    timeCalibrationOffset23 = cms.double(27.7),
    averageClusterSize = cms.double(1.5),
    averageEfficiency = cms.double(0.98),
    averageNoiseRate = cms.double(0.001), #intrinsic noise
#    numberOfStripsPerPartition = cms.int32(384),
    bxwidth = cms.int32(25),
    minBunch = cms.int32(-5), ## in terms of 25 ns
    maxBunch = cms.int32(3),
    inputCollection = cms.string('g4SimHitsMuonGEMHits'),
    digiModelString = cms.string('Simple'),
    digitizeOnlyMuons = cms.bool(False),
#    neutronGammaRoll = cms.vdouble(18., 23., 30., 39., 45., 52., 62., 76)#, #n and gamma bkg per roll
    neutronGammaRoll1 = cms.vdouble(69.3, 74.0, 101.7, 121.1, 145.5, 143.8, 199.1, 314.1), #n, gamma and charged prtcls bkg per roll of station1
    neutronGammaRoll2 = cms.vdouble(69.3, 74.0, 101.7, 121.1, 145.5, 143.8, 199.1, 314.1), #n, gamma and charged prtcls bkg per roll of station2
    neutronGammaRoll3 = cms.vdouble(69.3, 74.0, 101.7, 121.1, 145.5, 143.8, 199.1, 314.1, 314.1, 314.1, 314.1, 314.1), # bkg/roll of station3
    doNoiseCLS = cms.bool(True),
    minPabsNoiseCLS = cms.double(0.),
    simulateIntrinsicNoise = cms.bool(False),
    scaleLumi = cms.double(1.)
)
