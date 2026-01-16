import FWCore.ParameterSet.Config as cms

_common_BTLparameters = cms.PSet(
    BunchCrossingTime = cms.double(25),    # [ns]
    LightOutput       = cms.double(2285.), # [npe/MeV], including Light Yield, Light Collection Efficincy and Photon Detection Efficiency
    LCEpositionSlope  = cms.double(0.035)  # [1/cm] LCE variation vs longitudinal position shift
)

_barrel_MTDDigitizer = cms.PSet(
    digitizerName     = cms.string("BTLDigitizer"),
    inputSimHits      = cms.InputTag("g4SimHits:FastTimerHitsBarrel"),
    digiCollectionTag = cms.string("FTLBarrel"),
    maxSimHitsAccTime = cms.uint32(100),
    premixStage1      = cms.bool(False),
    premixStage1MinCharge = cms.double(1e-4),
    premixStage1MaxCharge = cms.double(1e6),
    DeviceSimulation = cms.PSet(
        _common_BTLparameters,
        LightCollectionSlope      = cms.double(0.0915), # [ns/cm]
        SigmaLightCollectionSlope = cms.double(0.001)  # [ns/cm] sigma of the light collection slope
        ),
    ElectronicsSimulation = cms.PSet(
        _common_BTLparameters,
        SigmaLCEpositionSlope     = cms.double(0.002),  # [1/cm] sigma of the LCE variation vs longitudinal position shift
        PulseT2Threshold          = cms.double(8.764),  # [uA] T2 threshold
        PulseEThrershold          = cms.double(20.32),  # [uA] energy threshold (it corresponds to 1 MeV)
        ChannelRearmMode          = cms.uint32(2),      # 0: the channel rearming is switched off
                                                        # 1: the channel is rearmed after the end-of-event signal
                                                        # 2: the channel is rearmed after ChannelRearmNClocks cycles of the TOFHiR clock
        ChannelRearmNClocks       = cms.double(3.),     # number of TOFHiR clock cycles after which the channel is rearmed
        T1Delay                   = cms.double(0.),     # [ns]
        SiPMGain                  = cms.double(9.389e5),# SiPM gain at Vov = 3 V
        PulseTbranchAParam        = cms.vdouble(-2.2, 1.89e-8), # average pulse amplitude in uA vs Gain * Npe in the TOFHiR's time branch
        PulseEbranchAParam        = cms.vdouble(-1.3, 1.01e-8), # average pulse amplitude in uA vs Gain * Npe in the TOFHiR's energy branch
        TimeAtThr1RiseParam       = cms.vdouble(1.9e6,-0.663), # time at threshold T1 (20 DAC) vs Gain * Npe on the rising edge
        TimeAtThr2RiseParam       = cms.vdouble(5.2e6,-0.704), # time at threshold T2 (28 DAC) vs Gain * Npe on the rising edge
        TimeOverThr1Param         = cms.vdouble(1.4776e9,7.93403,-3.78578e-10,5.42505e-18,-2.27325e-27,-7.32799e-10,12.933), # time over the T1 threshold vs Gain * Npe
        SmearTimeForOOTtails      = cms.bool(True),     # switch to turn ON/OFF the uncertainty due to photons from OOT hits
        ScintillatorRiseTime      = cms.double(1.1),    # [ns]
        ScintillatorDecayTime     = cms.double(42.8),   # [ns]
        StocasticParam            = cms.vdouble(14.746531, 0.7), # (0.030[ns]*7000^0.7, 0.7)
        DarkCountRate             = cms.double(0.),     # [GHz]
        DCRParam                  = cms.vdouble(50.583684, 0.41), # (0.034[ns]*6000/30^0.41, 0.41)
        SigmaElectronicNoise      = cms.double(0.420),  # [uA]
        SlewRateParam             = cms.vdouble(1.3e9,-3.5,10.9e-9,14.7), # parameterization of slew rate vs Gain * npe
        SigmaTDC                  = cms.double(0.0133), # [ns]
        SigmaClockGlobal          = cms.double(0.007),  # [ns], uncertainty due to the global LHC clock distribution
        SigmaClockRU              = cms.double(0.005),  # [ns], uncertainty due to clock distribution within the readout units
        CorrelationCoefficient    = cms.double(1.),     # correlation coefficient between T1 and T2 uncertainties

        PulseQParam               = cms.vdouble(-22.5, 0.0348), # pulse amplitude in ADC counts vs Npe
        PulseQResParam            = cms.vdouble(51., -0.88),    # relative amplitude resolution vs Npe
        )


)

_endcap_MTDDigitizer = cms.PSet(
    digitizerName     = cms.string("ETLDigitizer"),
    inputSimHits      = cms.InputTag("g4SimHits:FastTimerHitsEndcap"),
    digiCollectionTag = cms.string("FTLEndcap"),
    maxSimHitsAccTime = cms.uint32(100),
    premixStage1      = cms.bool(False),
    premixStage1MinCharge = cms.double(1e-4),
    premixStage1MaxCharge = cms.double(1e6),
    DeviceSimulation  = cms.PSet(
        bxTime               = cms.double(25),

        IntegratedLuminosity = cms.double(1000.0),
        FluenceVsRadius      = cms.string("1.937*TMath::Power(x,-1.706)"),
        LGADGainVsFluence    = cms.string("TMath::Min(15.,30.-x)"),
        LGADGainDegradation  = cms.string("TMath::Max(1.0, TMath::Min(x, x + 0.05/0.01 * (x - 1) + y * (1 - x)/0.01))"),
        applyDegradation     = cms.bool(False),
        meVPerMIP            = cms.double(0.085), #from HGCAL
        MPVMuon             = cms.string("1.21561e-05 + 8.89462e-07 / (x * x)"),
        MPVPion             = cms.string("1.24531e-05 + 7.16578e-07 / (x * x)"),
        MPVKaon             = cms.string("1.20998e-05 + 2.47192e-06 / (x * x * x)"),
        MPVElectron         = cms.string("1.30030e-05 + 1.55166e-07 / (x * x)"),
        MPVProton           = cms.string("1.13666e-05 + 1.20093e-05 / (x * x)"),
        tdcWindowStart      = cms.double(9.375), # now set to 3 x ETROC_clock, phase can be adjusted to set the start at any value
        tdcWindowEnd        = cms.double(21.875) # now set to tdcWindowStart + nominal TDC window width (12.5 ns), TDC window width can be adjusted.
        ),
    ElectronicsSimulation = cms.PSet(
        bxTime               = cms.double(25),
        IntegratedLuminosity = cms.double(1000.),      # [1/fb]
        # n bits for the ADC 
        adcNbits             = cms.uint32(8),
        # n bits for the TDC
        tdcNbits             = cms.uint32(11),
        # ADC saturation
        adcSaturation_MIP  = cms.double(100),
        # for different thickness
        adcThreshold_MIP   = cms.double(0.025),
        iThreshold_MIP     = cms.double(0.9525),
        # LSB for time of arrival estimate from TDC in ns
        toaLSB_ns          = cms.double(0.013),
        referenceChargeColl = cms.double(1.0),
        noiseLevel          = cms.double(0.1750),
        sigmaDistorsion     = cms.double(0.0),
        sigmaTDC            = cms.double(0.010),
        formulaLandauNoise  = cms.string("TMath::Max(0.020, 0.020 * (0.35 * (x - 1.0) + 1.0))") 
        )
)

from Configuration.Eras.Modifier_phase2_etlV4_cff import phase2_etlV4
phase2_etlV4.toModify(_endcap_MTDDigitizer.DeviceSimulation, meVPerMIP = 0.015 )

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
for _m in [_barrel_MTDDigitizer, _endcap_MTDDigitizer]:
    premix_stage1.toModify(_m, premixStage1 = True)

# Fast Timing
mtdDigitizer = cms.PSet( 
    accumulatorType   = cms.string("MTDDigiProducer"),
    makeDigiSimLinks  = cms.bool(False),
    verbosity         = cms.untracked.uint32(0),

    barrelDigitizer = _barrel_MTDDigitizer,
    endcapDigitizer = _endcap_MTDDigitizer
)
