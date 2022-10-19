import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcROCParameters_cfi import hgcROCSettings
from SimCalorimetry.HGCalSimAlgos.hgcSensorOpParams_cfi import hgcSiSensorIleak,hgcSiSensorCCE

# Base configurations for HGCal digitizers
eV_per_eh_pair = 3.62
fC_per_ele     = 1.6020506e-4
nonAgedCCEs    = [1.0, 1.0, 1.0]
nonAgedNoises  = [2100.0,2100.0,1600.0] #100,200,300 um (in electrons)
nonAgedNoises_v9 = [2000.0,2400.0,2000.0] # 120,200,300 um (in electrons)
thresholdTracksMIP = True

HGCAL_ileakParam_toUse    = cms.PSet(
    ileakParam = cms.vdouble( hgcSiSensorIleak('TDR_600V') )
)

HGCAL_cceParams_toUse = cms.PSet(
    cceParamFine  = cms.vdouble(hgcSiSensorCCE(120,'TDR_600V')),
    cceParamThin  = cms.vdouble(hgcSiSensorCCE(200,'TDR_600V')),
    cceParamThick = cms.vdouble(hgcSiSensorCCE(300,'TDR_600V')),
    )

HGCAL_noise_fC = cms.PSet(
    scaleByDose = cms.bool(False),
    scaleByDoseAlgo = cms.uint32(0),
    scaleByDoseFactor = cms.double(1),
    doseMap = cms.string(""),
    values = cms.vdouble( [x*fC_per_ele for x in nonAgedNoises] ), #100,200,300 um
    )

HFNose_noise_fC = HGCAL_noise_fC.clone()

HGCAL_noise_heback = cms.PSet(
    scaleByDose = cms.bool(False),
    scaleByDoseAlgo = cms.uint32(0),
    scaleByDoseFactor = cms.double(1),
    doseMap = cms.string(""), #empty dose map at begin-of-life
    sipmMap = cms.string(""), #if empty will prompt to use geometry-based definition
    referenceIdark = cms.double(-1),
    referenceXtalk = cms.double(-1),
    noise_MIP = cms.double(1./100.) #this is to be deprecated
)

HGCAL_chargeCollectionEfficiencies = cms.PSet(
    values = cms.vdouble( nonAgedCCEs )
    )

HGCAL_noises = cms.PSet(
    values = cms.vdouble([x for x in nonAgedNoises])
    )

# ECAL
hgceeDigitizer = cms.PSet(
    accumulatorType   = cms.string("HGCDigiProducer"),
    digitizer         = cms.string("HGCEEDigitizer"),
    hitsProducer      = cms.string("g4SimHits"),
    hitCollection     = cms.string("HGCHitsEE"),
    digiCollection    = cms.string("HGCDigisEE"),
    NoiseGeneration_Method = cms.bool(True),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    eVPerEleHolePair = cms.double(eV_per_eh_pair),
    tofDelay          = cms.double(-9),
    digitizationType  = cms.uint32(0),
    makeDigiSimLinks  = cms.bool(False),
    premixStage1      = cms.bool(False),
    premixStage1MinCharge = cms.double(0),
    premixStage1MaxCharge = cms.double(1e6),
    useAllChannels    = cms.bool(True),
    verbosity         = cms.untracked.uint32(0),
    digiCfg = cms.PSet(
        keV2fC           = cms.double(0.044259), #1000 eV/3.62 (eV per e) / 6.24150934e3 (e per fC)
        ileakParam       = cms.PSet(refToPSet_ = cms.string("HGCAL_ileakParam_toUse")),
        cceParams        = cms.PSet(refToPSet_ = cms.string("HGCAL_cceParams_toUse")),
        chargeCollectionEfficiencies = cms.PSet(refToPSet_ = cms.string("HGCAL_chargeCollectionEfficiencies")),
        noise_fC         = cms.PSet(refToPSet_ = cms.string("HGCAL_noise_fC")),
        doTimeSamples    = cms.bool(False),
        thresholdFollowsMIP = cms.bool(thresholdTracksMIP),
        feCfg   = hgcROCSettings.clone()
        )
    )

# HCAL front
hgchefrontDigitizer = cms.PSet(
    accumulatorType   = cms.string("HGCDigiProducer"),
    digitizer         = cms.string("HGCHEfrontDigitizer"),
    hitsProducer      = cms.string("g4SimHits"),
    hitCollection     = cms.string("HGCHitsHEfront"),
    digiCollection    = cms.string("HGCDigisHEfront"),
    NoiseGeneration_Method = cms.bool(True),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    tofDelay          = cms.double(-11),
    digitizationType  = cms.uint32(0),
    makeDigiSimLinks  = cms.bool(False),
    premixStage1      = cms.bool(False),
    premixStage1MinCharge = cms.double(0),
    premixStage1MaxCharge = cms.double(1e6),
    useAllChannels    = cms.bool(True),
    verbosity         = cms.untracked.uint32(0),
    digiCfg = cms.PSet(
        keV2fC           = cms.double(0.044259), #1000 eV / 3.62 (eV per e) / 6.24150934e3 (e per fC)
        ileakParam       = cms.PSet(refToPSet_ = cms.string("HGCAL_ileakParam_toUse")),
        cceParams        = cms.PSet(refToPSet_ = cms.string("HGCAL_cceParams_toUse")),
        chargeCollectionEfficiencies = cms.PSet(refToPSet_ = cms.string("HGCAL_chargeCollectionEfficiencies")),
        noise_fC         = cms.PSet(refToPSet_ = cms.string("HGCAL_noise_fC")),
        doTimeSamples    = cms.bool(False),
        thresholdFollowsMIP        = cms.bool(thresholdTracksMIP),
        feCfg   = hgcROCSettings.clone()
    )
)

# HCAL back
hgchebackDigitizer = cms.PSet(
    accumulatorType   = cms.string("HGCDigiProducer"),
    digitizer         = cms.string("HGCHEbackDigitizer"),
    hitsProducer      = cms.string("g4SimHits"),
    hitCollection     = cms.string("HGCHitsHEback"),
    digiCollection    = cms.string("HGCDigisHEback"),
    NoiseGeneration_Method = cms.bool(True),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    tofDelay          = cms.double(-14),
    digitizationType  = cms.uint32(1),
    makeDigiSimLinks  = cms.bool(False),
    premixStage1      = cms.bool(False),
    premixStage1MinCharge = cms.double(0),
    premixStage1MaxCharge = cms.double(1e6),
    useAllChannels    = cms.bool(True),
    verbosity         = cms.untracked.uint32(0),
    digiCfg = cms.PSet(
        #0 empty digitizer, 1 calice digitizer, 2 realistic digitizer
        algo          = cms.uint32(2),        
        noise         = cms.PSet(refToPSet_ = cms.string("HGCAL_noise_heback")), #scales both for scint raddam and sipm dark current
        keV2MIP       = cms.double(1./675.0),
        doTimeSamples = cms.bool(False),
        nPEperMIP = cms.double(21.0),
        nTotalPE  = cms.double(7500),        
        sdPixels  = cms.double(1e-6), # this is additional photostatistics noise (as implemented), not sure why it's here...
        thresholdFollowsMIP = cms.bool(thresholdTracksMIP),
        feCfg = hgcROCSettings.clone(
            adcNbits        = 10,      # standard ROC operations (was 2 bits more up to 11_0_0_pre12)
            adcSaturation_fC = 68.75,  # keep the adc LSB the same (i.e. set saturation one quarter value of pre12)
            tdcSaturation_fC  = 1000,  # allow up to 1000 MIPs as a max range, including ToA mode
            targetMIPvalue_ADC   = 15, # to be used for HGCROC gain proposal
            adcThreshold_fC = 0.5,     # unchanged with respect to pre12
            tdcOnset_fC       = 55,    # turn on TDC when 80% of the ADC range is reached (one quarter of pre12
            #                                        indicative at this point)
            tdcForToAOnset_fC = cms.vdouble(12.,12.,12.),  #turn ToA for 20% of the TDC threshold (indicative at this point)
        )
    )
)

# HFNose
hfnoseDigitizer = cms.PSet(
    accumulatorType   = cms.string("HGCDigiProducer"),
    digitizer         = cms.string("HFNoseDigitizer"),
    hitsProducer      = cms.string("g4SimHits"),
    hitCollection     = cms.string("HFNoseHits"),
    digiCollection    = cms.string("HFNoseDigis"),
    NoiseGeneration_Method = cms.bool(True),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    eVPerEleHolePair = cms.double(eV_per_eh_pair),
    tofDelay          = cms.double(-33),
    digitizationType  = cms.uint32(0),
    makeDigiSimLinks  = cms.bool(False),
    premixStage1      = cms.bool(False),
    premixStage1MinCharge = cms.double(0),
    premixStage1MaxCharge = cms.double(1e6),
    useAllChannels    = cms.bool(True),
    verbosity         = cms.untracked.uint32(0),
    digiCfg = cms.PSet(
        keV2fC           = cms.double(0.044259), #1000 eV/3.62 (eV per e) / 6.24150934e3 (e per fC)
        ileakParam       = cms.PSet(refToPSet_ = cms.string("HGCAL_ileakParam_toUse")),
        cceParams        = cms.PSet(refToPSet_ = cms.string("HGCAL_cceParams_toUse")),
        chargeCollectionEfficiencies = cms.PSet(refToPSet_ = cms.string("HGCAL_chargeCollectionEfficiencies")),
        noise_fC         = cms.PSet(refToPSet_ = cms.string("HFNose_noise_fC")),
        doTimeSamples    = cms.bool(False),
        thresholdFollowsMIP        = cms.bool(thresholdTracksMIP),
        feCfg   = hgcROCSettings.clone()
        )
    )

# this bypasses the noise simulation
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
for _m in [hgceeDigitizer, hgchefrontDigitizer, hgchebackDigitizer, hfnoseDigitizer]:
    premix_stage1.toModify(_m, premixStage1 = True)

#function to set noise to aged HGCal
endOfLifeCCEs = [0.5, 0.5, 0.7] # this is to be deprecated
endOfLifeNoises = [2400.0,2250.0,1750.0]  #this is to be deprecated
def HGCal_setEndOfLifeNoise(process,byDose=True,byDoseAlgo=0,byDoseAlgoSci=2,byDoseFactor=1):
    """
    includes all effects from radiation and gain choice
    (see also notes in HGCal_setRealisticStartupNoise)    
    """

    process=HGCal_setRealisticNoiseSi(process,byDose=byDose,byDoseAlgo=byDoseAlgo,byDoseFactor=byDoseFactor)
    process=HGCal_setRealisticNoiseSci(process,
                                       byDose=byDose,
                                       byDoseAlgo=byDoseAlgoSci,
                                       byDoseFactor=byDoseFactor,
                                       referenceIdark=0.25)
    return process

def HGCal_setEndOfLifeNoise_4000(process):
    process.HGCAL_cceParams_toUse = cms.PSet(
        cceParamFine  = cms.vdouble(hgcSiSensorCCE(120,'TDR_800V')),
        cceParamThin  = cms.vdouble(hgcSiSensorCCE(200,'TDR_800V')),
        cceParamThick = cms.vdouble(hgcSiSensorCCE(300,'TDR_800V')),
    )
    process.HGCAL_ileakParam_toUse    = cms.PSet(
        ileakParam = cms.vdouble(hgcSiSensorIleak('TDR_800V'))
    )
    return HGCal_setEndOfLifeNoise(process,byDoseFactor=1.333)

def HGCal_setEndOfLifeNoise_1500(process):
    process.HGCAL_cceParams_toUse = cms.PSet(
        cceParamFine  = cms.vdouble(hgcSiSensorCCE(120,'TDR_600V')),
        cceParamThin  = cms.vdouble(hgcSiSensorCCE(200,'TDR_600V')),
        cceParamThick = cms.vdouble(hgcSiSensorCCE(300,'TDR_600V')),
    )
    process.HGCAL_ileakParam_toUse    = cms.PSet(
        ileakParam = cms.vdouble(hgcSiSensorIleak('TDR_800V'))
    )
    return HGCal_setEndOfLifeNoise(process,byDoseFactor=0.5)

def HGCal_setRealisticStartupNoise(process):
    """ 
    include all effects except:
    * fluence impact on leakage current, CCE and SiPM dark current
    * dose impact on tile light yield
    dark current on SiPMs adjusted for a S/N ~ 7
    Notes
    * byDoseAlgo is used as a collection of bits to toggle: 
       * Si: FLUENCE, CCE, NOISE, PULSEPERGAIN, CACHEDOP (from lsb to Msb)
       * Sci: IGNORE_SIPMAREA, OVERRIDE_SIPMAREA, IGNORE_TILEAREA, IGNORE_DOSESCALE, IGNORE_FLUENCESCALE, IGNORE_NOISE, IGNORE_TILETYPE (from lsb to Msb)
      (for instance turning on the 0th  bit turns off the impact of fluence in Si)
    """
    process=HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=1)
    process=HGCal_setRealisticNoiseSci(process,byDose=True,byDoseAlgo=2+8+16,referenceIdark=0.125,referenceXtalk=0.01)
    return process

def HGCal_setRealisticStartupNoise_fixedSiPMTileAreasAndSN(process,targetSN=7,referenceXtalk=-1,ignorePedestal=False):
    """ 
    similar to HGCal_setRealisticStartupNoise but tile and SiPM areas are fixed
    as 4mm2 assumed use Idark=0.25 so that S/N ~ 7
    by changing the target S/N different the reference Idark will be scaled accordingly
    """
    process=HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=1)
    
    #scale dark current on the SiPM so that it corresponds to the target S/N
    idark=0.25/(targetSN/6.97)**2
    print('[HGCal_setRealisticStartupNoise_fixedSiPMTileAreasAndSN] for a target S/N={:3.2f} setting idark={:3.3f}nA'.format(targetSN,idark))
    process=HGCal_setRealisticNoiseSci(process,byDose=True,
                                       byDoseAlgo=2+4+8+16+64+128*ignorePedestal,
                                       byDoseSipmMap=cms.string("SimCalorimetry/HGCalSimProducers/data/sipmParams_all4mm2.txt"),
                                       referenceIdark=idark,
                                       referenceXtalk=referenceXtalk)
    return process


def HGCal_ignoreFluence(process):
    """
    include all effects except fluence impact on leakage current and CCE and SiPM dark current
    and dose impact on tile light yield
    (see also notes in HGCal_setRealisticStartupNoise)    
    """
    process=HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=1)
    process=HGCal_setRealisticNoiseSci(process,byDose=True,byDoseAlgo=2+8+16)
    return process

def HGCal_ignoreNoise(process):
    """
    include all effects except noise impact on leakage current and CCE, and scint
    (see also notes in HGCal_setRealisticStartupNoise)
    """
    process=HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=4)
    process=HGCal_setRealisticNoiseSci(process,byDose=True,byDoseAlgo=2+32)
    return process

def HGCal_ignorePulsePerGain(process):
    """
    include all effects except the per-gain pulse emulation
    for the moment this only done for Si
    (see also notes in HGCal_setRealisticStartupNoise)   
    """
    process=HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=8)
    process=HGCal_setRealisticNoiseSci(process,byDose=True,byDoseAlgo=2)
    return process

def HGCal_useCaching(process):
    """
    include all effects except cachine of siop parameters (gain cpu time)
    for the moment this only done for Si
    (see also notes in HGCal_setRealisticStartupNoise)        
    """    
    process=HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=16)
    process=HGCal_setRealisticNoiseSci(process,byDose=True,byDoseAlgo=2)
    return process

doseMap = cms.string("SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.7.20.txt")

def HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=0,byDoseMap=doseMap,byDoseFactor=1):
    process.HGCAL_noise_fC = cms.PSet(
        scaleByDose = cms.bool(byDose),
        scaleByDoseAlgo = cms.uint32(byDoseAlgo),
        scaleByDoseFactor = cms.double(byDoseFactor),
        doseMap = byDoseMap,
        values = cms.vdouble( [x*fC_per_ele for x in endOfLifeNoises] ), #100,200,300 um, to be deprecated
        )

    #this is to be deprecated
    process.HGCAL_chargeCollectionEfficiencies = cms.PSet(  
        values = cms.vdouble(endOfLifeCCEs)
        )
    
    #this is to be deprecated
    process.HGCAL_noises = cms.PSet(
        values = cms.vdouble([x for x in endOfLifeNoises])  
        )

    return process


def HFNose_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=0,byDoseMap=doseMap,byDoseFactor=1):
    process.HFNose_noise_fC = cms.PSet(
        scaleByDose = cms.bool(byDose),
        scaleByDoseAlgo = cms.uint32(byDoseAlgo),
        scaleByDoseFactor = cms.double(byDoseFactor),
        doseMap = byDoseMap,
        values = cms.vdouble( [x*fC_per_ele for x in endOfLifeNoises] ), #100,200,300 um
        )
    return process


def HGCal_setRealisticNoiseSci(process,
                               byDose=True,
                               byDoseAlgo=2,
                               byDoseMap=doseMap,
                               byDoseSipmMap=cms.string("SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt"),
                               byDoseFactor=1,
                               referenceIdark=0.25,
                               referenceXtalk=-1):
    process.HGCAL_noise_heback = cms.PSet(
        scaleByDose = cms.bool(byDose),
        scaleByDoseAlgo = cms.uint32(byDoseAlgo),
        scaleByDoseFactor = cms.double(byDoseFactor),
        doseMap = byDoseMap,
        sipmMap = byDoseSipmMap,
        referenceIdark = cms.double(referenceIdark),
        referenceXtalk = cms.double(referenceXtalk),
        noise_MIP = cms.double(1./5.), #this is to be deprecated (still needed for vanilla for the moment)
        )
    return process

def HGCal_disableNoise(process):
    process.HGCAL_noise_fC = cms.PSet(
        scaleByDose = cms.bool(False),
        scaleByDoseAlgo = cms.uint32(0),
        scaleByDoseFactor = cms.double(1),
        doseMap = cms.string(""),
        values = cms.vdouble(0,0,0), #100,200,300 um
    )
    process.HGCAL_noise_heback = cms.PSet(
        scaleByDose = cms.bool(False),
        scaleByDoseAlgo = cms.uint32(0),
        scaleByDoseFactor = cms.double(1),
        doseMap = cms.string(""),
        referenceIdark = cms.double(0.),
        referenceXtalk = cms.double(-1),
        noise_MIP = cms.double(0.), #zero noise (this is to be deprecated)
        )
    process.HGCAL_noises = cms.PSet(
        values = cms.vdouble(0,0,0)
    )
    return process

from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10

phase2_hgcalV10.toModify(HGCAL_noise_fC, values = [x*fC_per_ele for x in nonAgedNoises_v9])
phase2_hgcalV10.toModify(HGCAL_noises, values = [x for x in nonAgedNoises_v9])

def HFNose_setEndOfLifeNoise(process,byDose=True,byDoseAlgo=0,byDoseFactor=1):
    """includes all effects from radiation and gain choice"""
    # byDoseAlgo is used as a collection of bits to toggle: FLUENCE, CCE, NOISE, PULSEPERGAIN, CACHEDOP (from lsb to Msb)
    process=HFNose_setRealisticNoiseSi(process,byDose=byDose,byDoseAlgo=byDoseAlgo,byDoseMap=doseMapNose,byDoseFactor=byDoseFactor)
    return process

doseMapNose = cms.string("SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka_HFNose_3.7.20.12_Eta2.4.txt")
