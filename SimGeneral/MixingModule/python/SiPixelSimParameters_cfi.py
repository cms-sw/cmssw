import FWCore.ParameterSet.Config as cms

# This object is used to make configuration changes for different running
# scenarios, in this case for Run 2. See the code at the end of the
# SiPixelSimBlock definition.
from Configuration.StandardSequences.Eras import eras

def _modifyPixelDigitizerForRun2Bunchspacing25( digitizer ) :
    """
    Function that modifies the pixel digitiser for Run 2 with 25ns bunchspacing.
    First argument is the pixelDigitizer object.
    """
    # DynamicInefficency - 13TeV - 25ns case
    digitizer.theInstLumiScaleFactor = cms.double(364)
    digitizer.theLadderEfficiency_BPix1 = cms.vdouble( [1]*20 ) # this syntax makes an array with 20 copies of "1"
    digitizer.theLadderEfficiency_BPix2 = cms.vdouble( [1]*32 )
    digitizer.theLadderEfficiency_BPix3 = cms.vdouble( [1]*44 )
    digitizer.theModuleEfficiency_BPix1 = cms.vdouble( 1, 1, 1, 1, )
    digitizer.theModuleEfficiency_BPix2 = cms.vdouble( 1, 1, 1, 1, )
    digitizer.theModuleEfficiency_BPix3 = cms.vdouble( 1, 1, 1, 1 )
    digitizer.thePUEfficiency_BPix1 = cms.vdouble( 1.00023, -3.18350e-06, 5.08503e-10, -6.79785e-14 )
    digitizer.thePUEfficiency_BPix2 = cms.vdouble( 9.99974e-01, -8.91313e-07, 5.29196e-12, -2.28725e-15 )
    digitizer.thePUEfficiency_BPix3 = cms.vdouble( 1.00005, -6.59249e-07, 2.75277e-11, -1.62683e-15 )

def _modifyPixelDigitizerForRun2Bunchspacing50( digitizer ) :
    """
    Function that modifies the pixel digitiser for Run 2 with 50ns bunchspacing.
    
    First argument is the pixelDigitizer object.
    """
    # DynamicInefficency - 13TeV - 50ns case
    digitizer.theInstLumiScaleFactor = cms.double(246.4)
    digitizer.theLadderEfficiency_BPix1 = cms.vdouble( [0.979259,0.976677]*10 ) # This syntax makes a 20 element array of alternating numbers
    digitizer.theLadderEfficiency_BPix2 = cms.vdouble( [0.994321,0.993944]*16 )
    digitizer.theLadderEfficiency_BPix3 = cms.vdouble( [0.996787,0.996945]*22 )

def _modifyPixelDigitizerForPhase1Pixel( digitizer ) :
    """
    Function that modifies the pixel digitiser for the Phase 1 pixel detector.
    
    First argument is the pixelDigitizer object.
    """
    #use default switches, new analog respnse, d.k. 2/16
    #digitizer.MissCalibrate = False
    #digitizer.LorentzAngle_DB = False
    #digitizer.killModules = False
    #digitizer.useDB = False
    #digitizer.DeadModules_DB = False
    digitizer.NumPixelBarrel = cms.int32(4)
    digitizer.NumPixelEndcap = cms.int32(3)
    digitizer.ThresholdInElectrons_FPix = cms.double(2000.0)
    digitizer.ThresholdInElectrons_BPix = cms.double(2000.0)
    digitizer.ThresholdInElectrons_BPix_L1 = cms.double(2000.0)
    digitizer.FPix_SignalResponse_p0 = cms.double(0.00171)
    digitizer.FPix_SignalResponse_p1 = cms.double(0.711)
    digitizer.FPix_SignalResponse_p2 = cms.double(203.)
    digitizer.FPix_SignalResponse_p3 = cms.double(148.)
    digitizer.BPix_SignalResponse_p0 = cms.double(0.00171)
    digitizer.BPix_SignalResponse_p1 = cms.double(0.711)
    digitizer.BPix_SignalResponse_p2 = cms.double(203.)
    digitizer.BPix_SignalResponse_p3 = cms.double(148.)
    digitizer.thePixelColEfficiency_BPix1 = cms.double(0.999)
    digitizer.thePixelColEfficiency_BPix2 = cms.double(0.999)
    digitizer.thePixelColEfficiency_BPix3 = cms.double(0.999)
    digitizer.thePixelColEfficiency_BPix4 = cms.double(0.999)
    digitizer.thePixelEfficiency_BPix1 = cms.double(0.999)
    digitizer.thePixelEfficiency_BPix2 = cms.double(0.999)
    digitizer.thePixelEfficiency_BPix3 = cms.double(0.999)
    digitizer.thePixelEfficiency_BPix4 = cms.double(0.999)
    digitizer.thePixelChipEfficiency_BPix1 = cms.double(0.999)
    digitizer.thePixelChipEfficiency_BPix2 = cms.double(0.999)
    digitizer.thePixelChipEfficiency_BPix3 = cms.double(0.999)
    digitizer.thePixelChipEfficiency_BPix4 = cms.double(0.999)
    digitizer.thePixelColEfficiency_FPix1 = cms.double(0.999)
    digitizer.thePixelColEfficiency_FPix2 = cms.double(0.999)
    digitizer.thePixelColEfficiency_FPix3 = cms.double(0.999)
    digitizer.thePixelEfficiency_FPix1 = cms.double(0.999)
    digitizer.thePixelEfficiency_FPix2 = cms.double(0.999)
    digitizer.thePixelEfficiency_FPix3 = cms.double(0.999)
    digitizer.thePixelChipEfficiency_FPix1 = cms.double(0.999)
    digitizer.thePixelChipEfficiency_FPix2 = cms.double(0.999)
    digitizer.thePixelChipEfficiency_FPix3 = cms.double(0.999)
    # something broken in the configs above - turn off for now
    digitizer.AddPixelInefficiency = cms.bool(False)

def _modifyPixelDigitizerForPhase1PixelWithPileup( processObject ) :
    """
    Function that checks if there is pileup being used then modifies the
    pixel digitiser for the Phase 1 pixel detector accordingly.
    
    First argument is the "process" object. This function can only be applied
    with a <era>.makeProcessModifier() command, since it can only be applied
    at the end because the number of pileup interactions is not known yet.
    """
    if hasattr(processObject,'mix'): 
        n=0
        if hasattr(processObject.mix,'input'):
            n=processObject.mix.input.nbPileupEvents.averageNumber.value()
        if n>0:
            processObject.mix.digitizers.pixel.thePixelColEfficiency_BPix1 = cms.double(1.0-(0.0238*n/50.0))
            processObject.mix.digitizers.pixel.thePixelColEfficiency_BPix2 = cms.double(1.0-(0.0046*n/50.0))
            processObject.mix.digitizers.pixel.thePixelColEfficiency_BPix3 = cms.double(1.0-(0.0018*n/50.0))
            processObject.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(1.0-(0.0008*n/50.0))
            processObject.mix.digitizers.pixel.thePixelColEfficiency_FPix1 = cms.double(1.0-(0.0018*n/50.0))
            processObject.mix.digitizers.pixel.thePixelColEfficiency_FPix2 = cms.double(1.0-(0.0018*n/50.0))
            processObject.mix.digitizers.pixel.thePixelColEfficiency_FPix3 = cms.double(1.0-(0.0018*n/50.0))


SiPixelSimBlock = cms.PSet(
    DoPixelAging = cms.bool(False),
    ReadoutNoiseInElec = cms.double(350.0),
    deltaProductionCut = cms.double(0.03),
    RoutList = cms.vstring(
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
    AddPixelInefficiency = cms.bool(True),
    AddNoise = cms.bool(True),
    ChargeVCALSmearing = cms.bool(True),
    GainSmearing = cms.double(0.0),
    PixGeometryType = cms.string('idealForDigi'),                           
    useDB = cms.bool(False),
    LorentzAngle_DB = cms.bool(True),
    DeadModules_DB = cms.bool(True),
    killModules = cms.bool(True),
    NumPixelBarrel = cms.int32(3),
    NumPixelEndcap = cms.int32(2),
###    DeadModules = cms.VPSet()
)

#
# Apply the changes for the different Run 2 running scenarios
#
eras.run2_25ns_specific.toModify( SiPixelSimBlock, func=_modifyPixelDigitizerForRun2Bunchspacing25 )
eras.run2_50ns_specific.toModify( SiPixelSimBlock, func=_modifyPixelDigitizerForRun2Bunchspacing50 )
eras.phase1Pixel.toModify( SiPixelSimBlock, func=_modifyPixelDigitizerForPhase1Pixel )
# Note that this object must have a unique name, so I'll call it "modify<python filename>ForPhase1WithPileup_"
modifySimGeneralMixingModuleSiPixelSimParametersForPhase1WithPileup_ = eras.phase1Pixel.makeProcessModifier( _modifyPixelDigitizerForPhase1PixelWithPileup )

# Threshold in electrons are the Official CRAFT09 numbers:
# FPix(smearing)/BPix(smearing) = 2480(160)/2730(200)

#DEAD MODULES LIST: NEW LIST AFTER 2009 PIXEL REPAIRS
# https://twiki.cern.ch/twiki/bin/view/CMS/SiPixelQualityHistory
######Barrel
#Bad Module: 302055940 errorType 2 BadRocs=ff00
#Bad Module: 302059800 errorType 0 BadRocs=ffff
#BmI_SEC3_LYR2_LDR5F_MOD3 -- 302121992, "TBM-A"
#####Forward
#Bad Module: 352453892 errorType 0 BadRocs=ffff
#BmO_DISK1_BLD9_PNL2 -- 344014340, 344014344, 344014348
#Barrel 
#302187268, "none" (ROC 6) 

