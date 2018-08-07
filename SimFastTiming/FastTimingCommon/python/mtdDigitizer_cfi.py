import FWCore.ParameterSet.Config as cms

_barrel_MTDDigitizer = cms.PSet(
    digitizerName     = cms.string("BTLDigitizer"),
    inputSimHits      = cms.InputTag("g4SimHits:FastTimerHitsBarrel"),
    digiCollectionTag = cms.string("FTLBarrel"),
    maxSimHitsAccTime = cms.uint32(100),
    DeviceSimulation = cms.PSet(
        bxTime                   = cms.double(25),      # [ns] 
        LightYield               = cms.double(40000.),  # [photons/MeV]
        LightCollectionEff       = cms.double(0.30),
        LightCollectionTime      = cms.double(0.2),     # [ns]
        smearLightCollectionTime = cms.double(0.),      # [ns]
        PhotonDetectionEff       = cms.double(0.20),
        ),
    ElectronicsSimulation = cms.PSet(
        bxTime                     = cms.double(25),    # [ns] 
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
        Npe_to_pC                  = cms.double(0.016), # [pC] 
        Npe_to_V                   = cms.double(0.0064),# [V] 

        # n bits for the ADC 
        adcNbits          = cms.uint32(10),
        # n bits for the TDC
        tdcNbits          = cms.uint32(10),
        # ADC saturation (in pC)
        adcSaturation_MIP = cms.double(600.),
        # for different thickness
        adcThreshold_MIP   = cms.double(0.064),
        # LSB for time of arrival estimate from TDC in ns
        toaLSB_ns         = cms.double(0.020),
        )
    )

_endcap_MTDDigitizer = cms.PSet(
    digitizerName     = cms.string("ETLDigitizer"),
    inputSimHits      = cms.InputTag("g4SimHits:FastTimerHitsEndcap"),
    digiCollectionTag = cms.string("FTLEndcap"),
    maxSimHitsAccTime = cms.uint32(100),
    DeviceSimulation  = cms.PSet(
        bxTime            = cms.double(25),
        tofDelay          = cms.double(1),
        meVPerMIP         = cms.double(0.085), # from HGCal
        ),
    ElectronicsSimulation = cms.PSet(
        bxTime             = cms.double(25),
        etaResolution      = cms.string("0.03+0.0025*x"), # This is just a dummy dependence on eta.
        # n bits for the ADC 
        adcNbits           = cms.uint32(12),
        # n bits for the TDC
        tdcNbits           = cms.uint32(12),
        # ADC saturation
        adcSaturation_MIP  = cms.double(102),
        # for different thickness
        adcThreshold_MIP   = cms.double(0.025),
        # LSB for time of arrival estimate from TDC in ns
        toaLSB_ns          = cms.double(0.005),
        )
    )

# Fast Timing
mtdDigitizer = cms.PSet( 
    accumulatorType   = cms.string("MTDDigiProducer"),
    makeDigiSimLinks  = cms.bool(False),
    verbosity         = cms.untracked.uint32(0),

    barrelDigitizer = _barrel_MTDDigitizer,    
    endcapDigitizer = _endcap_MTDDigitizer
)

