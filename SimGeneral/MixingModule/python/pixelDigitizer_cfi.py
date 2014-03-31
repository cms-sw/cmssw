import FWCore.ParameterSet.Config as cms

pixelDigitizer = cms.PSet(
    accumulatorType = cms.string("SiPixelDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True),
    ReadoutNoiseInElec = cms.double(350.0),
    DeltaProductionCut = cms.double(0.03),
    ROUList = cms.vstring(
        'TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof'),
    OffsetSmearing = cms.double(0.0),
    ThresholdInElectrons_FPix = cms.double(3000.0), 
    ThresholdInElectrons_BPix = cms.double(3500.0),
    ThresholdInElectrons_BPix_L1 = cms.double(3500.0),
    AddThresholdSmearing = cms.bool(True),
    ThresholdSmearing_FPix = cms.double(210.0),
    ThresholdSmearing_BPix = cms.double(245.0),
    ThresholdSmearing_BPix_L1 = cms.double(245.0),
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
    AddPixelInefficiency = cms.int32(0), # deprecated, use next option
    AddPixelInefficiencyFromPython = cms.bool(True),
    AddNoise = cms.bool(True),
    ChargeVCALSmearing = cms.bool(True),
    GainSmearing = cms.double(0.0),
    GeometryType = cms.string('idealForDigi'),                           
    useDB = cms.bool(True),
    LorentzAngle_DB = cms.bool(True),
    DeadModules_DB = cms.bool(True),
##    killModules = cms.bool(False),
##    DeadModules_DB = cms.bool(False),
    killModules = cms.bool(True),
    NumPixelBarrel = cms.int32(3),
    NumPixelEndcap = cms.int32(2),
    thePixelColEfficiency_BPix1 = cms.double(1.0), 	# Only used when AddPixelInefficiency = true
    thePixelColEfficiency_BPix2 = cms.double(1.0),
    thePixelColEfficiency_BPix3 = cms.double(1.0),
    thePixelColEfficiency_FPix1 = cms.double(0.999),
    thePixelColEfficiency_FPix2 = cms.double(0.999),
    thePixelEfficiency_BPix1 = cms.double(1.0), 	# Only used when AddPixelInefficiency = true
    thePixelEfficiency_BPix2 = cms.double(1.0),
    thePixelEfficiency_BPix3 = cms.double(1.0),
    thePixelEfficiency_FPix1 = cms.double(0.999),
    thePixelEfficiency_FPix2 = cms.double(0.999),
    thePixelChipEfficiency_BPix1 = cms.double(1.0), 	# Only used when AddPixelInefficiency = true
    thePixelChipEfficiency_BPix2 = cms.double(1.0),
    thePixelChipEfficiency_BPix3 = cms.double(1.0),
    thePixelChipEfficiency_FPix1 = cms.double(0.999),
    thePixelChipEfficiency_FPix2 = cms.double(0.999),
    theLadderEfficiency_BPix1 = cms.vdouble(
        0.978351,
        0.971877,
        0.974283,
        0.969328,
        0.972922,
        0.970964,
        0.975762,
        0.974786,
        0.980244,
        0.978452,
        0.982129,
        0.979737,
        0.984381,
        0.983971,
        0.98186,
        0.983283,
        0.981485,
        0.979753,
        0.980287,
        0.975195
        ),
    theLadderEfficiency_BPix2 = cms.vdouble(
        0.996276,
        0.993354,
        0.993752,
        0.992948,
        0.993871,
        0.992317,
        0.997733,
        0.992516,
        0.992649,
        0.993425,
        0.994065,
        0.993481,
        0.993169,
        0.994223,
        0.992397,
        0.99509,
        0.995177,
        0.995319,
        0.994925,
        0.992933,
        0.994111,
        0.9948,
        0.994711,
        0.994294,
        0.995392,
        0.994229,
        0.994414,
        0.995271,
        0.993585,
        0.995264,
        0.992977,
        0.993642
        ),
    theLadderEfficiency_BPix3 = cms.vdouble(
        0.996206,
        0.998039,
        0.995801,
        0.99665,
        0.996414,
        0.995755,
        0.996518,
        0.995584,
        0.997171,
        0.998056,
        0.99595,
        0.997473,
        0.996858,
        0.996486,
        0.997442,
        0.998002,
        0.995429,
        0.997939,
        0.996896,
        0.997434,
        0.996616,
        0.996439,
        0.996546,
        0.997597,
        0.995435,
        0.996396,
        0.99621,
        0.998316,
        0.998431,
        0.99598,
        0.997063,
        0.996245,
        0.997502,
        0.996254,
        0.997545,
        0.997553,
        0.996722,
        0.996107,
        0.996588,
        0.996277,
        0.99785,
        0.997087,
        0.998139,
        0.997139
        ),
    theModuleEfficiency_BPix1 = cms.vdouble(
        1.00361,
        0.999371,
        0.961242,
        0.766481
        ),
    theModuleEfficiency_BPix2 = cms.vdouble(
        1.00069,
        0.999792,
        0.99562,
        0.955627
        ),
    theModuleEfficiency_BPix3 = cms.vdouble(
        1.00006,
        0.999744,
        0.998147,
        1.00314
        ),
    thePUEfficiency_BPix1 = cms.vdouble(
        1.0181,
        -2.28345e-07,
        -1.30042e-09
        ),
    thePUEfficiency_BPix2 = cms.vdouble(
        1.00648,
        -1.28515e-06,
        -1.85915e-10
        ),
    thePUEfficiency_BPix3 = cms.vdouble(
        1.0032,
        -1.96206e-08,
        -1.99009e-10
        ),
DeadModules = cms.VPSet(
 cms.PSet(Dead_detID = cms.int32(302055940), Module = cms.string("tbmB"))
,cms.PSet(Dead_detID = cms.int32(302059800), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302121992), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302123296), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302125060), Module = cms.string("tbmA"))
,cms.PSet(Dead_detID = cms.int32(302125076), Module = cms.string("tbmA"))
,cms.PSet(Dead_detID = cms.int32(302126364), Module = cms.string("tbmB"))
,cms.PSet(Dead_detID = cms.int32(302126596), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302127136), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302188552), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302188824), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302194200), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302195232), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302197252), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(302197784), Module = cms.string("whole"))
##forward
,cms.PSet(Dead_detID = cms.int32(352453892), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352453896), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352453900), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352453904), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352454916), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352454920), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352454924), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352454928), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352455940), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352455944), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352455948), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352455952), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352454148), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352454152), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352454156), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352455172), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352455176), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352455180), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352456196), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352456200), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(352456204), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(343999748), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(343999752), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(343999756), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(343999760), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344014340), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344014344), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344014348), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344019460), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344019464), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344019468), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344077572), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344077576), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344077580), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344077584), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344078596), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344078600), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344078604), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344078608), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344079620), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344079624), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344079628), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344079632), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344078852), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344078856), Module = cms.string("whole"))
,cms.PSet(Dead_detID = cms.int32(344078860), Module = cms.string("whole"))
#,cms.PSet(Dead_detID = cms.int32(302187268), Module = cms.string("none"))
#,cms.PSet(Dead_detID = cms.int32(302195472), Module = cms.string("none"))
#,cms.PSet(Dead_detID = cms.int32(302128136), Module = cms.string("none"))
)

###    DeadModules = cms.VPSet()
)

# Threshold in electrons are the Official CRAFT09 numbers:
# FPix(smearing)/BPix(smearing) = 2480(160)/2730(200)

#DEAD MODULES LIST: NEW LIST AFTER 2009 PIXEL REPAIRS
# https://twiki.cern.ch/twiki/bin/view/CMS/SiPixelQualityHistory

######Barrel
#Bad Module: 302055940 errorType 2 BadRocs=ff00
#Bad Module: 302059800 errorType 0 BadRocs=ffff
#Bad Module: 302121992 errorType 0 BadRocs=ffff
#BmI_SEC3_LYR2_LDR5F_MOD3 -- 302121992, "TBM-A"
#Bad Module: 302123296 errorType 0 BadRocs=ffff
#BpO_SEC1_LYR2_LDR1H_MOD4 -- 302123296, "whole"
#Bad Module: 302125060 errorType 1 BadRocs=ff
#Bad Module: 302125076 errorType 1 BadRocs=ff
#BpO_SEC4_LYR2_LDR8F_MOD1 -- 302125076, "TBM-A"
#Bad Module: 302126364 errorType 2 BadRocs=ff00
#BpO_SEC7_LYR2_LDR13F_MOD3 -- 302126364, "TBM-B"
#Bad Module: 302126596 errorType 0 BadRocs=ffff
#BmO_SEC7_LYR2_LDR14F_MOD4 -- 302126596, "whole"
#Bad Module: 302127136 errorType 0 BadRocs=ffff
#BpO_SEC8_LYR2_LDR16H_MOD4 -- 302127136, "whole"
#Bad Module: 302188552 errorType 0 BadRocs=ffff
#BmI_SEC2_LYR3_LDR4F_MOD3 -- 302188552, "whole"
#Bad Module: 302188824 errorType 0 BadRocs=ffff
#Bad Module: 302194200 errorType 0 BadRocs=ffff
#Bad Module: 302195232 errorType 0 BadRocs=ffff
#BpI_SEC8_LYR3_LDR22H_MOD4 -- 302195232, "whole"
#Bad Module: 302197252 errorType 0 BadRocs=ffff
#Bad Module: 302197784 errorType 0 BadRocs=ffff
#BpI_SEC5_LYR3_LDR12F_MOD2 -- 302197784, "whole"

#####Forward
#Bad Module: 352453892 errorType 0 BadRocs=ffff
#Bad Module: 352453896 errorType 0 BadRocs=ffff
#Bad Module: 352453900 errorType 0 BadRocs=ffff
#Bad Module: 352453904 errorType 0 BadRocs=ffff
#Bad Module: 352454916 errorType 0 BadRocs=ffff
#Bad Module: 352454920 errorType 0 BadRocs=ffff
#Bad Module: 352454924 errorType 0 BadRocs=ffff
#Bad Module: 352454928 errorType 0 BadRocs=ffff
#Bad Module: 352455940 errorType 0 BadRocs=ffff
#Bad Module: 352455944 errorType 0 BadRocs=ffff
#Bad Module: 352455948 errorType 0 BadRocs=ffff
#Bad Module: 352455952 errorType 0 BadRocs=ffff
#Bad Module: 352454148 errorType 0 BadRocs=ffff
#Bad Module: 352454152 errorType 0 BadRocs=ffff
#Bad Module: 352454156 errorType 0 BadRocs=ffff
#Bad Module: 352455172 errorType 0 BadRocs=ffff
#Bad Module: 352455176 errorType 0 BadRocs=ffff
#Bad Module: 352455180 errorType 0 BadRocs=ffff
#Bad Module: 352456196 errorType 0 BadRocs=ffff
#Bad Module: 352456200 errorType 0 BadRocs=ffff
#Bad Module: 352456204 errorType 0 BadRocs=ffff
#Bad Module: 343999748 errorType 0 BadRocs=ffff
#Bad Module: 343999752 errorType 0 BadRocs=ffff
#Bad Module: 343999756 errorType 0 BadRocs=ffff
#Bad Module: 343999760 errorType 0 BadRocs=ffff
#Bad Module: 344014340 errorType 0 BadRocs=ffff
#Bad Module: 344014344 errorType 0 BadRocs=ffff
#Bad Module: 344014348 errorType 0 BadRocs=ffff
#BmO_DISK1_BLD9_PNL2 -- 344014340, 344014344, 344014348
#Bad Module: 344019460 errorType 0 BadRocs=ffff
#Bad Module: 344019464 errorType 0 BadRocs=ffff
#Bad Module: 344019468 errorType 0 BadRocs=ffff
#BmI_DISK1_BLD11_PNL2 -- 344019460, 344019464, 344019468
#Bad Module: 344077572 errorType 0 BadRocs=ffff
#Bad Module: 344077576 errorType 0 BadRocs=ffff
#Bad Module: 344077580 errorType 0 BadRocs=ffff
#Bad Module: 344077584 errorType 0 BadRocs=ffff
#Bad Module: 344078596 errorType 0 BadRocs=ffff
#Bad Module: 344078600 errorType 0 BadRocs=ffff
#Bad Module: 344078604 errorType 0 BadRocs=ffff
#Bad Module: 344078608 errorType 0 BadRocs=ffff
#Bad Module: 344079620 errorType 0 BadRocs=ffff
#Bad Module: 344079624 errorType 0 BadRocs=ffff
#Bad Module: 344079628 errorType 0 BadRocs=ffff
#Bad Module: 344079632 errorType 0 BadRocs=ffff
#Bad Module: 344078852 errorType 0 BadRocs=ffff
#Bad Module: 344078856 errorType 0 BadRocs=ffff
#Bad Module: 344078860 errorType 0 BadRocs=ffff

#Barrel 
#302187268, "none" (ROC 6) 
#302195472, "none" (ROC 0)
#302128136, "none" (ROC 3)

