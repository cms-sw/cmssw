import FWCore.ParameterSet.Config as cms
###The following configurations for efficiency loss are read from the DB by default
###If you want to set private inefficiency factors use this file and change the appropriate values
###The values below are set for 2012 running conditions
###For the PhaseI detector please change the SLHCUpgradeSimulations/Configuration/python/phase1TkCustoms.py
def customise_pixel_ineff(process):
    process.mix.digitizers.pixel.theInstLumiScaleFactor = cms.double(221.95)
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix1 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix2 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix3 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelColEfficiency_FPix1 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelColEfficiency_FPix2 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix1 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix2 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix3 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelEfficiency_FPix1 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_FPix2 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix1 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix2 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix3 = cms.double(1.0)
    process.mix.digitizers.pixel.thePixelChipEfficiency_FPix1 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_FPix2 = cms.double(0.999)
    process.mix.digitizers.pixel.theLadderEfficiency_BPix1 = cms.vdouble(
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
        )
    process.mix.digitizers.pixel.theLadderEfficiency_BPix2 = cms.vdouble(
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
        )
    process.mix.digitizers.pixel.theLadderEfficiency_BPix3 = cms.vdouble(
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
        )
    process.mix.digitizers.pixel.theModuleEfficiency_BPix1 = cms.vdouble(
        1.00361,
        0.999371,
        0.961242,
        0.953582 
        )
    process.mix.digitizers.pixel.theModuleEfficiency_BPix2 = cms.vdouble(
        1.00069,
        0.999792,
        0.99562,
        1.00341
        )
    process.mix.digitizers.pixel.theModuleEfficiency_BPix3 = cms.vdouble(
        1.00006,
        0.999744,
        0.998147,
        1.00039
        )
    process.mix.digitizers.pixel.thePUEfficiency_BPix1 = cms.vdouble(
        1.0181,
        -2.28345e-07,
        -1.30042e-09
        )
    process.mix.digitizers.pixel.thePUEfficiency_BPix2 = cms.vdouble(
        1.00648,
        -1.28515e-06,
        -1.85915e-10
        )
    process.mix.digitizers.pixel.thePUEfficiency_BPix3 = cms.vdouble(
        1.0032,
        -1.96206e-08,
        -1.99009e-10
        )
    process.mix.digitizers.pixel.theInnerEfficiency_FPix1 = cms.double(1.0)
    process.mix.digitizers.pixel.theInnerEfficiency_FPix2 = cms.double(1.0)
    process.mix.digitizers.pixel.theOuterEfficiency_FPix1 = cms.double(1.0)
    process.mix.digitizers.pixel.theOuterEfficiency_FPix2 = cms.double(1.0)
    process.mix.digitizers.pixel.thePUEfficiency_FPix_Inner = cms.vdouble(1.0)
    process.mix.digitizers.pixel.thePUEfficiency_FPix_Outer = cms.vdouble(1.0)
    return process
