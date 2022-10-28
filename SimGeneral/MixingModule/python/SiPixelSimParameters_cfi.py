import FWCore.ParameterSet.Config as cms

# This object is used to make configuration changes for different running
# scenarios, in this case for Run 2. See the code at the end of the
# SiPixelSimBlock definition.

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
    digitizer.ThresholdInElectrons_BPix_L1 = cms.double(3000.0)
    digitizer.ThresholdInElectrons_BPix_L2 = cms.double(2600.0)
    digitizer.FPix_SignalResponse_p0 = cms.double(0.00171)
    digitizer.FPix_SignalResponse_p1 = cms.double(0.711)
    digitizer.FPix_SignalResponse_p2 = cms.double(203.)
    digitizer.FPix_SignalResponse_p3 = cms.double(148.)
    digitizer.BPix_SignalResponse_p0 = cms.double(0.00171)
    digitizer.BPix_SignalResponse_p1 = cms.double(0.711)
    digitizer.BPix_SignalResponse_p2 = cms.double(203.)
    digitizer.BPix_SignalResponse_p3 = cms.double(148.)
    # gains and offsets are ints in the Clusterizer, so round to the same value
    digitizer.ElectronsPerVcal           = cms.double(47)   # L2-4: 47  +- 4.7
    digitizer.ElectronsPerVcal_L1        = cms.double(50)   # L1:   49.6 +- 2.6
    digitizer.ElectronsPerVcal_Offset    = cms.double(-60)  # L2-4: -60 +- 130
    digitizer.ElectronsPerVcal_L1_Offset = cms.double(-670) # L1:   -670 +- 220
    digitizer.UseReweighting = cms.bool(True)
    digitizer.KillBadFEDChannels = cms.bool(True)

def _modifyPixelDigitizerForRun3( digitizer ):

    digitizer.ThresholdInElectrons_FPix = cms.double(1600.0)
    digitizer.ThresholdInElectrons_BPix = cms.double(1600.0)
    digitizer.ThresholdInElectrons_BPix_L1 = cms.double(2000.0)
    digitizer.ThresholdInElectrons_BPix_L2 = cms.double(1600.0)

SiPixelSimBlock = cms.PSet(
    SiPixelQualityLabel = cms.string(''),
    KillBadFEDChannels = cms.bool(False),
    UseReweighting = cms.bool(False),
    applyLateReweighting = cms.bool(False),
    store_SimHitEntryExitPoints = cms.bool(False),
    PrintClusters = cms.bool(False),
    PrintTemplates = cms.bool(False),
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
    ThresholdInElectrons_BPix_L2 = cms.double(3500.0),
    AddThresholdSmearing = cms.bool(True),
    ThresholdSmearing_FPix = cms.double(210.0),
    ThresholdSmearing_BPix = cms.double(245.0),
    ThresholdSmearing_BPix_L1 = cms.double(245.0),
    ThresholdSmearing_BPix_L2 = cms.double(245.0),
    NoiseInElectrons = cms.double(175.0),
    MissCalibrate = cms.bool(True),
    MissCalInLateCR = cms.bool(True),
    FPix_SignalResponse_p0 = cms.double(0.0043),
    FPix_SignalResponse_p1 = cms.double(1.31),
    FPix_SignalResponse_p2 = cms.double(93.6),
    FPix_SignalResponse_p3 = cms.double(134.6),
    BPix_SignalResponse_p0 = cms.double(0.0035),
    BPix_SignalResponse_p1 = cms.double(1.23),
    BPix_SignalResponse_p2 = cms.double(97.4),
    BPix_SignalResponse_p3 = cms.double(126.5),
    ElectronsPerVcal = cms.double(65.5),
    ElectronsPerVcal_L1 = cms.double(65.5),
    ElectronsPerVcal_Offset = cms.double(-414.0),
    ElectronsPerVcal_L1_Offset = cms.double(-414.0),
    ElectronPerAdc = cms.double(135.0),
    TofUpperCut = cms.double(12.5),
    AdcFullScale = cms.int32(255),
    AdcFullScLateCR = cms.int32(255),
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

# activate charge reweighing for 2016 pixel detector (UL 2016)
from Configuration.Eras.Modifier_pixel_2016_cff import pixel_2016
pixel_2016.toModify(SiPixelSimBlock,UseReweighting=True)

#
# Apply the changes for the different Run 2 running scenarios
#
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify( SiPixelSimBlock, func=_modifyPixelDigitizerForPhase1Pixel )

# use Label 'forDigitizer' for years >= 2018
from CalibTracker.SiPixelESProducers.SiPixelQualityESProducer_cfi import siPixelQualityESProducer
from Configuration.Eras.Modifier_run2_SiPixel_2018_cff import run2_SiPixel_2018
run2_SiPixel_2018.toModify(siPixelQualityESProducer,siPixelQualityLabel = 'forDigitizer',)
run2_SiPixel_2018.toModify(SiPixelSimBlock, SiPixelQualityLabel = 'forDigitizer',)

# change the digitizer threshold for Run3
# - new layer1 installed: expected improvement in timing alignment of L1 and L2
# - update the rest of the detector to 1600e 

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(SiPixelSimBlock, func=_modifyPixelDigitizerForRun3)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(SiPixelSimBlock,
    AddNoise = True,
    AddNoisyPixels = False,
    AddPixelInefficiency = False, #done in second step
    KillBadFEDChannels = False, #done in second step
)

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

