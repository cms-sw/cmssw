import FWCore.ParameterSet.Config as cms

# Module to create simulated GEM digis and analyze them in flight.
simMuonGEMDigis = cms.EDProducer("GEMDigiProducer",
    signalPropagationSpeed = cms.double(0.66),
    cosmics = cms.bool(False),
    timeResolution = cms.double(5),
    timeJitter = cms.double(1.0),
    averageShapingTime = cms.double(50.0),
    timeCalibrationOffset = cms.double(19.9),
    averageClusterSize = cms.double(1.5),
    averageEfficiency = cms.double(0.98),
    averageNoiseRate = cms.double(0.0),#intrinsic noise
    numberOfStripsPerPartition = cms.int32(384),
    bxwidth = cms.int32(25),
    minBunch = cms.int32(-5), ## in terms of 25 ns
    maxBunch = cms.int32(3),
    inputCollection = cms.string('g4SimHitsMuonGEMHits'),
    digiModelString = cms.string('Simple'),#might be Trivial, Simple or SimpleAnalyzeInFlight
    digitizeOnlyMuons = cms.bool(False),# digitize all the particles or only muons
    cutElecMomentum = cms.double(0.01),# cut those non muon particles with the momentum < than cutElecMomentum [GeV/c] to avoid the time issues
    cutForCls = cms.int32(3),#minimal distance [#strips] between the centres of the fired clusters of the muon and the secondary particle
    neutronGammaRoll1 = cms.double(76.),# combined effective rate in Hz/cm^2 for the first eta partition                     
    neutronGammaRoll2 = cms.double(62.),                    
    neutronGammaRoll3 = cms.double(52.),                     
    neutronGammaRoll4 = cms.double(45.),                    
    neutronGammaRoll5 = cms.double(39.),                     
    neutronGammaRoll6 = cms.double(30.),                  
    neutronGammaRoll7 = cms.double(23.),                    
    neutronGammaRoll8 = cms.double(18.)                    
)
