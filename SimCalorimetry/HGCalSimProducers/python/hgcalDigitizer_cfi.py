import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HGCalSimProducers.hgcROCParameters_cfi import hgcROCSettings

# Base configurations for HGCal digitizers
eV_per_eh_pair = 3.62
fC_per_ele     = 1.6020506e-4
nonAgedCCEs    = [1.0, 1.0, 1.0]
nonAgedNoises  = [2100.0,2100.0,1600.0] #100,200,300 um (in electrons)
nonAgedNoises_v9 = [2000.0,2400.0,2000.0] # 120,200,300 um (in electrons)
thresholdTracksMIP = True

ileakParam_600V     = [0.993,-42.668]
ileakParam_800V     = [0.996,-42.464]
HGCAL_ileakParam_toUse    = cms.PSet(
    ileakParam = cms.vdouble(ileakParam_600V)
    )

#  line+log tdr 600V
cceParamFine_tdr600  = [1.5e+15, -3.00394e-17, 0.318083]      #120
cceParamThin_tdr600  = [1.5e+15, -3.09878e-16, 0.211207]      #200
cceParamThick_tdr600 = [6e+14,   -7.96539e-16, 0.251751]      #300
#  line+log tdr 800V
cceParamFine_tdr800  = [4.2e+15, 2.35482e-18,  0.553187]      #120
cceParamThin_tdr800  = [1.5e+15, -1.98109e-16, 0.280567]      #200
cceParamThick_tdr800 = [6e+14,   -5.24999e-16, 0.357616]      #300
#  line+log ttu 600V
cceParamFine_ttu600  = [1.5e+15,  9.98631e-18, 0.343774]      #120
cceParamThin_ttu600  = [1.5e+15, -2.17083e-16, 0.304873]      #200
cceParamThick_ttu600 = [6e+14,   -8.01557e-16, 0.157375]      #300
#  line+log ttu 800V
cceParamFine_ttu800  = [1.5e+15, 3.35246e-17,  0.251679]      #120
cceParamThin_ttu800  = [1.5e+15, -1.62096e-16, 0.293828]      #200
cceParamThick_ttu800 = [6e+14,   -5.95259e-16, 0.183929]      #300
#  line+log tdr 600V EPI
cceParamFine_epi600  = [3.5e+15, -9.73872e-19, 0.263812]      #100
cceParamThin_epi600  = [1.5e+15, -3.09878e-16, 0.211207]      #200
cceParamThick_epi600 = [6e+14,   -7.96539e-16, 0.251751]      #300

HGCAL_cceParams_toUse = cms.PSet(
    cceParamFine  = cms.vdouble(cceParamFine_tdr600),
    cceParamThin  = cms.vdouble(cceParamThin_tdr600),
    cceParamThick = cms.vdouble(cceParamThick_tdr600)
    )

HGCAL_noise_fC = cms.PSet(
    scaleByDose = cms.bool(False),
    scaleByDoseAlgo = cms.uint32(0),
    doseMap = cms.string(""),
    values = cms.vdouble( [x*fC_per_ele for x in nonAgedNoises] ), #100,200,300 um
    )

HGCAL_noise_heback = cms.PSet(
    scaleByDose = cms.bool(False),
    scaleByDoseAlgo = cms.uint32(0),
    doseMap = cms.string(""), #empty dose map at begin-of-life
    noise_MIP = cms.double(1./100.)
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
    hitCollection     = cms.string("HGCHitsEE"),
    digiCollection    = cms.string("HGCDigisEE"),
    NoiseGeneration_Method = cms.bool(True),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    eVPerEleHolePair = cms.double(eV_per_eh_pair),
    tofDelay          = cms.double(5),
    geometryType      = cms.uint32(0),
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
    hitCollection  = cms.string("HGCHitsHEfront"),
    digiCollection = cms.string("HGCDigisHEfront"),
    NoiseGeneration_Method = cms.bool(True),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    tofDelay          = cms.double(5),
    geometryType      = cms.uint32(0),
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
    hitCollection = cms.string("HcalHits"),
    digiCollection = cms.string("HGCDigisHEback"),
    NoiseGeneration_Method = cms.bool(True),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    tofDelay          = cms.double(1),
    geometryType      = cms.uint32(0),
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
        scaleByTileArea= cms.bool(False),
        scaleBySipmArea= cms.bool(False),
        sipmMap       = cms.string("SimCalorimetry/HGCalSimProducers/data/sipmParams_geom-10.txt"),
        noise         = cms.PSet(refToPSet_ = cms.string("HGCAL_noise_heback")), #scales both for scint raddam and sipm dark current
        keV2MIP       = cms.double(1./675.0),
        doTimeSamples = cms.bool(False),
        nPEperMIP = cms.double(21.0),
        nTotalPE  = cms.double(7500),
        xTalk     = cms.double(0.01),
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
    hitCollection     = cms.string("HFNoseHits"),
    digiCollection    = cms.string("HFNoseDigis"),
    NoiseGeneration_Method = cms.bool(True),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    eVPerEleHolePair = cms.double(eV_per_eh_pair),
    tofDelay          = cms.double(5),
    geometryType      = cms.uint32(1),
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
        thresholdFollowsMIP        = cms.bool(thresholdTracksMIP),
        feCfg   = hgcROCSettings.clone()
        )
    )

# this bypasses the noise simulation
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
for _m in [hgceeDigitizer, hgchefrontDigitizer, hgchebackDigitizer, hfnoseDigitizer]:
    premix_stage1.toModify(_m, premixStage1 = True)

#function to set noise to aged HGCal
endOfLifeCCEs = [0.5, 0.5, 0.7]
endOfLifeNoises = [2400.0,2250.0,1750.0]
def HGCal_setEndOfLifeNoise(process,byDose=True,byDoseAlgo=0):
    """includes all effects from radiation and gain choice"""
    process=HGCal_setRealisticNoiseSi(process,byDose=byDose,byDoseAlgo=byDoseAlgo)
    process=HGCal_setRealisticNoiseSci(process,byDose=byDose,byDoseAlgo=byDoseAlgo)
    return process

def HGCal_setRealisticStartupNoise(process):
    """include all effects except fluence impact on leakage current and CCE"""
    #note: realistic electronics with Sci is not yet switched on
    process=HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=1)
    return process

def HGCal_setRealisticNoiseSi(process,byDose=True,byDoseAlgo=0):
    process.HGCAL_noise_fC = cms.PSet(
        scaleByDose = cms.bool(byDose),
        scaleByDoseAlgo = cms.uint32(byDoseAlgo),
        doseMap = cms.string("SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.5.15.9.txt"),
        values = cms.vdouble( [x*fC_per_ele for x in endOfLifeNoises] ), #100,200,300 um
        )
    process.HGCAL_chargeCollectionEfficiencies = cms.PSet(
        values = cms.vdouble(endOfLifeCCEs)
        )
    process.HGCAL_noises = cms.PSet(
        values = cms.vdouble([x for x in endOfLifeNoises])
        )
    return process

def HGCal_setRealisticNoiseSci(process,byDose=True,byDoseAlgo=0):
    process.HGCAL_noise_heback = cms.PSet(
        scaleByDose = cms.bool(byDose),
        scaleByDoseAlgo = cms.uint32(byDoseAlgo),
        doseMap = cms.string("SimCalorimetry/HGCalSimProducers/data/doseParams_3000fb_fluka-3.5.15.9.txt"),
        noise_MIP = cms.double(1./5.) #uses noise map
        )
    return process



def HGCal_disableNoise(process):
    process.HGCAL_noise_fC = cms.PSet(
        values = cms.vdouble(0,0,0), #100,200,300 um
    )
    process.HGCAL_noise_heback = cms.PSet(
        scaleByDose = cms.bool(False),
        scaleByDoseAlgo = cms.uint32(0),
        doseMap = cms.string(""),
        noise_MIP = cms.double(0.) #zero noise
        )
    process.HGCAL_noises = cms.PSet(
        values = cms.vdouble(0,0,0)
    )
    return process

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9

phase2_hgcalV9.toModify( hgceeDigitizer,
      geometryType      = cms.uint32(1),
)
phase2_hgcalV9.toModify( hgchefrontDigitizer,
      geometryType      = cms.uint32(1),
)
phase2_hgcalV9.toModify( hgchebackDigitizer,
      geometryType      = cms.uint32(1),
      hitCollection = cms.string("HGCHitsHEback"),
)
phase2_hgcalV9.toModify(HGCAL_noise_fC, values = [x*fC_per_ele for x in nonAgedNoises_v9])
phase2_hgcalV9.toModify(HGCAL_noises, values = [x for x in nonAgedNoises_v9])
