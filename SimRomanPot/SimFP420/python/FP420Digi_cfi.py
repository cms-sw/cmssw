import FWCore.ParameterSet.Config as cms

FP420Digi = cms.EDProducer("DigitizerFP420",
    VerbosityLevel = cms.untracked.int32(0),    ## = -50
    ROUList = cms.vstring('g4SimHitsFP420SI'),  ##
    NumberFP420Detectors = cms.int32(3),        ## =3 means 2 Trackers: +FP420 and -FP420; =0 -> no FP420 at all 
    NumberFP420Stations = cms.int32(3),         ## means 2 Stations w/ arm 8m
    NumberFP420SPlanes = cms.int32(6),          ## means 5 SuperPlanes
    NoFP420Noise = cms.bool(False),             ##
    ElectronFP420PerAdc = cms.double(300.0),    ## 
    zFP420 = cms.double(420000.0),              ##
    zFP420D3 = cms.double(8000.0),              ##
    zFP420D2 = cms.double(4000.0),              ##
    FedFP420Algorithm = cms.int32(1),           ## 2
    FedFP420LowThreshold = cms.double(6.0),     ## 4.
    FedFP420HighThreshold = cms.double(6.5),    ## 4.5
    AdcFP420Threshold = cms.double(6.0),        ##   5.0
    AddNoisyPixelsFP420 = cms.bool(True),       ##
    LowtofCutFP420 = cms.double(1390.0),        ##
    ApplyTofCutFP420 = cms.bool(True),          ## False
    ApplyChargeIneffFP420 = cms.bool(False),    ## True
    NumberHPS240Detectors = cms.int32(3),         ## =3 means 2 Trackers: +HPS240 and -HPS240; =0 -> no HPS240 at all 
    NumberHPS240Stations = cms.int32(3),          ## means 2 Stations w/ arm 8m
    NumberHPS240SPlanes = cms.int32(6),           ## means 5 SuperPlanes
    NoHPS240Noise = cms.bool(False),              ##
    ElectronHPS240PerAdc = cms.double(300.0),     ## 
    zHPS240 = cms.double(240000.0),               ##
    zHPS240D3 = cms.double(8000.0),               ##
    zHPS240D2 = cms.double(4000.0),               ##
    FedHPS240Algorithm = cms.int32(1),            ## 2
    FedHPS240LowThreshold = cms.double(6.0),      ## 4.
    FedHPS240HighThreshold = cms.double(6.5),     ## 4.5
    AdcHPS240Threshold = cms.double(6.0),         ##   5.0
    AddNoisyPixelsHPS240 = cms.bool(True),        ##
    LowtofCutHPS240 = cms.double(790.0),         ##
    ApplyTofCutHPS240 = cms.bool(True),           ## False
    ApplyChargeIneffHPS240 = cms.bool(False)      ## True
) 
