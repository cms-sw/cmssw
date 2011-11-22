import FWCore.ParameterSet.Config as cms

simSiPixelDigis = cms.EDProducer("SiPixelDigitizer",
    ReadoutNoiseInElec = cms.double(350.0),
    DeltaProductionCut = cms.double(0.03),
    ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
        'g4SimHitsTrackerHitsPixelBarrelHighTof', 
        'g4SimHitsTrackerHitsPixelEndcapLowTof', 
        'g4SimHitsTrackerHitsPixelEndcapHighTof'),
    OffsetSmearing = cms.double(0.0),
    ThresholdInElectrons_FPix = cms.double(2870.0), 
    ThresholdInElectrons_BPix = cms.double(3700.0),
    ThresholdInElectrons_BPix_L1 = cms.double(3700.0),
    AddThresholdSmearing = cms.bool(True),
    ThresholdSmearing_FPix = cms.double(210.0),
    ThresholdSmearing_BPix = cms.double(410.0),
    NoiseInElectrons = cms.double(175.0),
    MissCalibrate = cms.bool(True),
    FPix_SignalResponse_p0 = cms.double(0.0043),
    FPix_SignalResponse_p1 = cms.double(1.31),
    FPix_SignalResponse_p2 = cms.double(93.6),
    FPix_SignalResponse_p3 = cms.double(134.6),
    BPix_SignalResponse_p0 = cms.double(0.0035),
    BPix_SignalResponse_p1 = cms.double(1.23),
    BPix_SignalResponse_p2 = cms.double(97.4),
    BPix_SignalResponse_p3 = cms.double(126.5),
    ElectronsPerVcal = cms.double(65.5),
    ElectronsPerVcal_Offset = cms.double(-414.0),
    ElectronPerAdc = cms.double(135.0),
    TofUpperCut = cms.double(12.5),
    AdcFullScale = cms.int32(255),
    AdcFullScaleStack = cms.int32(255),
    FirstStackLayer = cms.int32(5),
    TofLowerCut = cms.double(-12.5),
    TanLorentzAnglePerTesla_FPix = cms.double(0.106),
    TanLorentzAnglePerTesla_BPix = cms.double(0.106),
    AddNoisyPixels = cms.bool(True),
    Alpha2Order = cms.bool(True),
    AddPixelInefficiency = cms.int32(0),
    AddNoise = cms.bool(True),
    ChargeVCALSmearing = cms.bool(True),
    GainSmearing = cms.double(0.0),
    GeometryType = cms.string('idealForDigi'),                           
    useDB = cms.bool(False),
    LorentzAngle_DB = cms.bool(True),
    DeadModules_DB = cms.bool(False),
    killModules = cms.bool(True),
    DeadModules = cms.VPSet(),
    NumPixelBarrel = cms.int32(9),
    NumPixelEndcap = cms.int32(2),
    PixelEff     = cms.double(1.0),
    PixelColEff  = cms.double(1.0),
    PixelChipEff = cms.double(1.0)
)


# Threshold in electrons are the Official CRAFT09 numbers:
# FPix(smearing)/BPix(smearing) = 2480(160)/2730(200)

#DEAD MODULES LIST: NEW LIST AFTER 2009 PIXEL REPAIRS
# https://twiki.cern.ch/twiki/bin/view/CMS/SiPixelQualityHistory

#Barrel 
#BpI_SEC5_LYR3_LDR12F_MOD2 -- 302197784, "whole" 
#BpI_SEC8_LYR3_LDR22H_MOD4 -- 302195232, "whole" 
#BpO_SEC1_LYR2_LDR1H_MOD4 -- 302123296, "whole" 
#BpO_SEC8_LYR2_LDR16H_MOD4 -- 302127136, "whole" 
#BpO_SEC4_LYR2_LDR8F_MOD1 -- 302125076, "TBM-A" 
#BpO_SEC7_LYR2_LDR13F_MOD3 -- 302126364, "TBM-B" 
#BmI_SEC2_LYR3_LDR4F_MOD3 -- 302188552, "whole" 
#BmI_SEC3_LYR2_LDR5F_MOD3 -- 302121992, "TBM-A" 
#BmO_SEC7_LYR2_LDR14F_MOD4 -- 302126596, "whole" 
#302187268, "none" (ROC 6) 
#302195472, "none" (ROC 0)
#302128136, "none" (ROC 3)

#Forward 
#BmO_DISK1_BLD9_PNL2 -- 344014340, 344014344, 344014348
#BmI_DISK1_BLD11_PNL2 -- 344019460, 344019464, 344019468

#    DeadModules = cms.VPSet(cms.PSet(Dead_detID = cms.int32(302197784), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302195232), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302123296), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302127136), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302125076), Module = cms.string("tbmA")),
#                            cms.PSet(Dead_detID = cms.int32(302126364), Module = cms.string("tbmB")),
#                            cms.PSet(Dead_detID = cms.int32(302188552), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302121992), Module = cms.string("tbmA")),
#                            cms.PSet(Dead_detID = cms.int32(302126596), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344014340), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344014344), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344014348), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344019460), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344019464), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344019468), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302187268), Module = cms.string("none")),
#                            cms.PSet(Dead_detID = cms.int32(302195472), Module = cms.string("none")),
#                            cms.PSet(Dead_detID = cms.int32(302128136), Module = cms.string("none")))



