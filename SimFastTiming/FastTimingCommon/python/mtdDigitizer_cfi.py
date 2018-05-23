import FWCore.ParameterSet.Config as cms

_barrel_MTDDigitizer = cms.PSet(
    digitizerName     = cms.string("BTLDigitizer"),
    inputSimHits      = cms.InputTag("g4SimHits:FastTimerHitsBarrel"),
    digiCollectionTag = cms.string("FTLBarrel"),
    maxSimHitsAccTime = cms.uint32(100),
    DeviceSimulation = cms.PSet(
        bxTime            = cms.double(25),
        tofDelay          = cms.double(1),        
        meVPerMIP         = cms.double(3.438), # 0.9 * (4mm of LYSO * 9.55 MeV/cm) to adjust to MVP rather than average ionization loss
        ),
    ElectronicsSimulation = cms.PSet(
        # n bits for the ADC 
        adcNbits          = cms.uint32(10),
        # n bits for the TDC
        tdcNbits          = cms.uint32(10),
        # ADC saturation
        adcSaturation_MIP  = cms.double(102),
        # for different thickness
        adcThreshold_MIP   = cms.double(0.025),
        # LSB for time of arrival estimate from TDC in ns
        toaLSB_ns         = cms.double(0.005),
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
        # n bits for the ADC 
        adcNbits          = cms.uint32(12),
        # n bits for the TDC
        tdcNbits          = cms.uint32(12),
        # ADC saturation
        adcSaturation_MIP  = cms.double(102),
        # for different thickness
        adcThreshold_MIP   = cms.double(0.025),
        # LSB for time of arrival estimate from TDC in ns
        toaLSB_ns         = cms.double(0.005),
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

