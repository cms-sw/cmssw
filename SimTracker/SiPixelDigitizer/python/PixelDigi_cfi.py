import FWCore.ParameterSet.Config as cms

simSiPixelDigis = cms.EDFilter("SiPixelDigitizer",
    ReadoutNoiseInElec = cms.double(500.0),
    DeltaProductionCut = cms.double(0.03),
    ROUList = cms.vstring('TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof'),
    OffsetSmearing = cms.double(0.0),
    NoiseInElectrons = cms.double(175.0),
    ThresholdInElectrons = cms.double(2500.0),
    MissCalibrate = cms.bool(True),
    TofUpperCut = cms.double(12.5),
    ElectronPerAdc = cms.double(135.0),
    AdcFullScale = cms.int32(255),
    TofLowerCut = cms.double(-12.5),
    TanLorentzAnglePerTesla = cms.double(0.106),
    AddNoisyPixels = cms.bool(True),
    Alpha2Order = cms.bool(True),
    AddPixelInefficiency = cms.int32(0),
    AddNoise = cms.bool(True),
    GainSmearing = cms.double(0.0)
)



