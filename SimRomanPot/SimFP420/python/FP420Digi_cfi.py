import FWCore.ParameterSet.Config as cms

FP420Digi = cms.EDProducer("DigitizerFP420",
    VerbosityLevel = cms.untracked.int32(0),
    ROUList = cms.vstring('g4SimHitsFP420SI'),
    NumberFP420Stations = cms.int32(3),
    NumberFP420Detectors = cms.int32(3),
    NumberFP420SPlanes = cms.int32(6),
    NoFP420Noise = cms.bool(False),
    ElectronFP420PerAdc = cms.double(300.0),
    z420 = cms.double(420000.0),
    zD3 = cms.double(8000.0),
    zD2 = cms.double(4000.0),
    FedFP420Algorithm = cms.int32(1),
    FedFP420LowThreshold = cms.double(6.0),
    FedFP420HighThreshold = cms.double(6.5),
    AdcFP420Threshold = cms.double(6.0),
    AddNoisyPixels = cms.bool(True),
    LowtofCutAndTo200ns = cms.double(1350.0),
    ApplyTofCut = cms.bool(True)
)



