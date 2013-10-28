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
    digiModelString = cms.string('SimpleAnalyzeInFlight'),
    digitizeOnlyMuons = cms.bool(False),
    cutElecMomentum = cms.double(0.01),
    cutForCls = cms.int32(3),
    neutronGammaRoll1 = cms.double(76),#314.1 combined effective rate in Hz/cm^2 for the first eta partition                     
    neutronGammaRoll2 = cms.double(62), # 199.1                 
    neutronGammaRoll3 = cms.double(52), #   143.8                 
    neutronGammaRoll4 = cms.double(45), #  145.5                
    neutronGammaRoll5 = cms.double(39), #   121.1                 
    neutronGammaRoll6 = cms.double(30), #    101.7             
    neutronGammaRoll7 = cms.double(23), #     74.0              
    neutronGammaRoll8 = cms.double(18) #      69.3             
)
