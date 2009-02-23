import FWCore.ParameterSet.Config as cms

simSiPixelDigis = cms.EDFilter("SiPixelDigitizer",
    ReadoutNoiseInElec = cms.double(500.0),
    DeltaProductionCut = cms.double(0.03),
    ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof', 
        'g4SimHitsTrackerHitsPixelBarrelHighTof', 
        'g4SimHitsTrackerHitsPixelEndcapLowTof', 
        'g4SimHitsTrackerHitsPixelEndcapHighTof'),
    OffsetSmearing = cms.double(0.0),
    ThresholdInElectrons_FPix = cms.double(2870.0), 
    ThresholdInElectrons_BPix = cms.double(3700.0),
    AddThresholdSmearing = cms.bool(True),
    ThresholdSmearing_FPix = cms.double(200.0),
    ThresholdSmearing_BPix = cms.double(410.0),
    NoiseInElectrons = cms.double(175.0),
    MissCalibrate = cms.bool(True),
    ElectronsPerVcal = cms.double(65.5),
    ElectronsPerVcal_Offset = cms.double(-414.0),
    ElectronPerAdc = cms.double(135.0),
    TofUpperCut = cms.double(12.5),
    AdcFullScale = cms.int32(255),
    TofLowerCut = cms.double(-12.5),
    TanLorentzAnglePerTesla_FPix = cms.double(0.106),
    TanLorentzAnglePerTesla_BPix = cms.double(0.106),
    AddNoisyPixels = cms.bool(True),
    Alpha2Order = cms.bool(True),
    AddPixelInefficiency = cms.int32(0),
    AddNoise = cms.bool(True),
    GainSmearing = cms.double(0.0),
    GeometryType = cms.string('idealForDigi'),                           
    useDB = cms.bool(False),
    LorentzAngle_DB = cms.bool(True),
    DeadModules_DB = cms.bool(True),
    killModules = cms.bool(True),
    DeadModules = cms.VPSet()
)


# Threshold in electrons are the Official pixel numbers since january 2009: D. Kotlinski
#    ThresholdInElectrons_FPix = cms.double(2870.0) 
#    ThresholdInElectrons_BPix = cms.double(3700.0)

#DEAD MODULES LIST:
#    DeadModules = cms.VPSet(cms.PSet(Dead_detID = cms.int32(302197784), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302195232), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302123296), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302127136), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302125076), Module = cms.string("tbmA")),
#                            cms.PSet(Dead_detID = cms.int32(302126364), Module = cms.string("tbmB")),
#                            cms.PSet(Dead_detID = cms.int32(302188552), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(302121992), Module = cms.string("tbmA")),
#                            cms.PSet(Dead_detID = cms.int32(302126596), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344074500), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344074504), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344074508), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344074512), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344074756), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344074760), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344074764), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344075524), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344075528), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344075532), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344075536), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344075780), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344075784), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344075788), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344076548), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344076552), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344076556), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344076560), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344076804), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344076808), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344076812), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344005128), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344020236), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344020240), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344020488), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344020492), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344019212), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344019216), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344019464), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344019468), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344018188), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344018192), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344018440), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344018444), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344014340), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344014344), Module = cms.string("whole")),
#                            cms.PSet(Dead_detID = cms.int32(344014348), Module = cms.string("whole")))


#List of dead pixel modules:

#Barrel: D. Kotlinski
#
#No HV
#-----
#BpI5 - L3, FED12,chan 36, 16 rocs, BpI_SEC5_LYR3_LDR12F_MOD2  whole module
#BpI8 - L3, FED15,chan 26, 1/2 module 8 rocs  BpI_SEC8_LYR3_LDR22H_MOD4  whole module
#BpO1 - L2, FED7, chan 20, 1/2 module 8 rocs  BpO_SEC1_LYR2_LDR1H_MOD4  whole module
#BpO8 - L2, FED8, chan  5, 1/2 module 8 rocs  BpO_SEC8_LYR2_LDR16H_MOD4 whole module
#
#Other
#-----
#BpO4 - L2, FED4,ch 15, bad ROC header,1/2 of a full module,8 rocs BpO_SEC4_LYR2_LDR8F_MOD1  TBM-A
#BpO7 - L2, FED9,ch15, token lost, 1/2 of a full module,8 rocs BpO_SEC7_LYR2_LDR13F_MOD3 TBM-B
#BmI2 - L3, FED22,ch32, token lost, full module,16 rocs BmI_SEC2_LYR3_LDR4F_MOD3  whole module
#BmI3 - L2, FED21,ch3, bad ROC header,1/2 of full module, 8 rocs BmI_SEC3_LYR2_LDR5F_MOD3  TBM-A
#BmO7 - FED30, chan.3,7, module dead, 16 rocs BmO_SEC7_LYR2_LDR14F_MOD4 whole module
#
#whole module - all ROCs should be disabled
#TBM-A        - ROCs 0 to 7 should be disabled
#TBM-B        - ROCs 8 to 15 should be disabled

#Forward: P. Merkel
#
#Dead sector: BmO_D2 blades 4 to 6
#344074500
#       04
#       08
#       12
#344074756
#       60
#       64
#344075524
#       28
#       32
#       36
#344075780
#       84
#       88
#344076548
#       52
#       56
#       60
#344076804
#       08
#       12
#
#No HV in plaquette BmI_D1_BLD1_PNL2_PLQ2 (plaquette = module) -> detID 344005128

#Forward plaquettes with no bias voltage: K. Ecklund (sept. 9th 2008)
#
#FPix_BmI_D1_BLD10_PNL1_PLQ3  detID=344020236
#FPix_BmI_D1_BLD10_PNL1_PLQ4  detID=.......40
#FPix_BmI_D1_BLD10_PNL2_PLQ2  detID=....20488
#FPix_BmI_D1_BLD10_PNL2_PLQ3  detID=.......92
#FPix_BmI_D1_BLD11_PNL1_PLQ3  detID=....19212
#FPix_BmI_D1_BLD11_PNL1_PLQ4  detID=.......16
#FPix_BmI_D1_BLD11_PNL2_PLQ2  detID=....19464
#FPix_BmI_D1_BLD11_PNL2_PLQ3  detID=.......68
#FPix_BmI_D1_BLD12_PNL1_PLQ3  detID=....18188
#FPix_BmI_D1_BLD12_PNL1_PLQ4  detID=.......92
#FPix_BmI_D1_BLD12_PNL2_PLQ2  detID=....18440
#FPix_BmI_D1_BLD12_PNL2_PLQ3  detID=.......44

#Forward panel with bad baseline: L. Uplegger (sept. 15th 2008)
#
#FPix_BmO_D1_BLD9_PNL2   detID=344014340, 344014344 and 344014348.

#BPix dead ROCS: are not removed yet from the Digitizer.
#
#BPix_BmI_SEC3_LYR3_LDR9F_MOD4_ROC6
#BPix_BmI_SEC8_LYR3_LDR21F_MOD1_ROC0
#BPix_BmI_SEC7_LYR2_LDR13F_MOD3_ROC3
