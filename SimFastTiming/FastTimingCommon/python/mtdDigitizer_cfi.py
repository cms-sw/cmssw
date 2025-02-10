import FWCore.ParameterSet.Config as cms

_common_BTLparameters = cms.PSet(
    bxTime                   = cms.double(25),      # [ns]
    LightYield               = cms.double(40000.),  # [photons/MeV]
    LightCollectionEff       = cms.double(0.25),
    PhotonDetectionEff       = cms.double(0.20),
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
        LightCollectionSlope     = cms.double(0.075),   # [ns/cm]
        LCEpositionSlope         = cms.double(0.071),   # [1/cm] LCE variation vs longitudinal position shift
        ),
    ElectronicsSimulation = cms.PSet(
        _common_BTLparameters,
        TestBeamMIPTimeRes         = cms.double(4.293), # This is given by 0.048[ns]*sqrt(8000.), in order to
                                                        # rescale the time resolution of 1 MIP = 8000 p.e.
        ScintillatorRiseTime       = cms.double(1.1),   # [ns]
        ScintillatorDecayTime      = cms.double(40.),   # [ns]
        ChannelTimeOffset          = cms.double(0.),    # [ns]
        smearChannelTimeOffset     = cms.double(0.),    # [ns]
        EnergyThreshold            = cms.double(4.),    # [photo-electrons]
        TimeThreshold1             = cms.double(20.),   # [photo-electrons]
        TimeThreshold2             = cms.double(50.),   # [photo-electrons]
        ReferencePulseNpe          = cms.double(100.),  # [photo-electrons]
        DarkCountRate              = cms.double(10.),   # [GHz]
        SinglePhotonTimeResolution = cms.double(0.060), # [ns]
        SigmaElectronicNoise       = cms.double(1.),    # [p.e.]
        SigmaClock                 = cms.double(0.015), # [ns]
        CorrelationCoefficient     = cms.double(1.),
        SmearTimeForOOTtails       = cms.bool(True),
        Npe_to_pC                  = cms.double(0.016), # [pC]
        Npe_to_V                   = cms.double(0.0064),# [V]
        SigmaRelTOFHIRenergy       = cms.vdouble(0.139,-4.35e-05,3.315e-09,-1.20e-13,1.67e-18), # [%] coefficients of 4th degree Chebyshev polynomial parameterization

        # n bits for the ADC 
        adcNbits          = cms.uint32(10),
        # n bits for the TDC
        tdcNbits          = cms.uint32(10),
        # ADC saturation
        adcSaturation_MIP = cms.double(600.),           # [pC]
        # for different thickness
        adcThreshold_MIP   = cms.double(0.064),         # [pC]
        # LSB for time of arrival estimate from TDC
        toaLSB_ns         = cms.double(0.020),          # [ns]
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
        tofDelay             = cms.double(1),
        meVPerMIP            = cms.double(0.085), #from HGCAL
        MPVMuon             = cms.string("1.21561e-05 + 8.89462e-07 / (x * x)"),
        MPVPion             = cms.string("1.24531e-05 + 7.16578e-07 / (x * x)"),
        MPVKaon             = cms.string("1.20998e-05 + 2.47192e-06 / (x * x * x)"),
        MPVElectron         = cms.string("1.30030e-05 + 1.55166e-07 / (x * x)"),
        MPVProton           = cms.string("1.13666e-05 + 1.20093e-05 / (x * x)")
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
