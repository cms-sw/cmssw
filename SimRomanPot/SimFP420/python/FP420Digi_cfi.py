import FWCore.ParameterSet.Config as cms

FP420Digi = cms.EDFilter("DigitizerFP420",
    z420 = cms.double(420000.0),
    NumberFP420Stations = cms.int32(3),
    VerbosityLevel = cms.untracked.int32(0),
    ROUList = cms.vstring('g4SimHitsFP420SI'),
    NumberFP420Detectors = cms.int32(3),
    NoFP420Noise = cms.bool(False),
    ElectronFP420PerAdc = cms.double(300.0),
    FedFP420HighThreshold = cms.double(4.5),
    zD3 = cms.double(8000.0),
    zD2 = cms.double(4000.0),
    LowtofCutAndTo200ns = cms.double(1350.0),
    FedFP420LowThreshold = cms.double(4.0),
    AdcFP420Threshold = cms.double(5.0),
    AddNoisyPixels = cms.bool(True),
    NumberFP420SPlanes = cms.int32(6),
    ApplyTofCut = cms.bool(True),
    FedFP420Algorithm = cms.int32(1)
)



