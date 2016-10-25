import FWCore.ParameterSet.Config as cms

# Base configurations for HGCal digitizers
eV_per_eh_pair = 3.62
fC_per_ele     = 1.6020506e-4
nonAgedNoises = [2100.0,2100.0,1600.0] #100,200,300 um (in electrons)

# ECAL
hgceeDigitizer = cms.PSet( 
    accumulatorType   = cms.string("HGCDigiProducer"),
    hitCollection     = cms.string("HGCHitsEE"),
    digiCollection    = cms.string("HGCDigisEE"),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    eVPerEleHolePair = cms.double(eV_per_eh_pair),
    tofDelay          = cms.double(1),
    digitizationType  = cms.uint32(0),
    makeDigiSimLinks  = cms.bool(False),
    useAllChannels    = cms.bool(True),
    verbosity         = cms.untracked.uint32(0),
    digiCfg = cms.PSet( 
        keV2fC           = cms.double(0.044259), #1000 eV/3.62 (eV per e) / 6.24150934e3 (e per fC)
        noise_fC         = cms.vdouble( [x*fC_per_ele for x in nonAgedNoises] ), #100,200,300 um
        doTimeSamples    = cms.bool(False),                                         
        feCfg   = cms.PSet( 
            # 0 only ADC, 1 ADC with pulse shape, 2 ADC+TDC with pulse shape
            fwVersion         = cms.uint32(2),
            # leakage to bunches -2, -1, in-time, +1, +2, +3 (from J. Kaplon)
            #NOTE: this is a fixed-size array inside the simulation (for speed) change accordingly!
            adcPulse          = cms.vdouble(0.00, 0.017,   0.817,   0.163,  0.003,  0.000), 
            pulseAvgT         = cms.vdouble(0.00, 23.42298,13.16733,6.41062,5.03946,4.5320), 
            # n bits for the ADC 
            adcNbits          = cms.uint32(10),
            # ADC saturation
            adcSaturation_fC  = cms.double(100),
            # the tdc resolution smearing (in picoseconds)
            tdcResolutionInPs = cms.double( 0.001 ),
            # LSB for TDC, assuming 12 bit dynamic range to 10 pC
            tdcNbits          = cms.uint32(12),
            # TDC saturation
            tdcSaturation_fC  = cms.double(10000),
            # raise threshold flag (~MIP/2) this is scaled 
            # for different thickness
            adcThreshold_fC   = cms.double(0.672),
            # raise usage of TDC and mode flag (from J. Kaplon)
            tdcOnset_fC       = cms.double(60) ,
            # LSB for time of arrival estimate from TDC in ns
            toaLSB_ns         = cms.double(0.005),
            #toa computation mode (0=by weighted energy, 1=simple threshold)
            toaMode           = cms.uint32(1),
            # TDC charge drain parameterisation (from J. Kaplon)
            tdcChargeDrainParameterisation = cms.vdouble(
                -919.13, 365.36, -14.10, 0.2,
                 -21.85, 49.39,  22.21,  0.8,
                 -0.28,   27.14,  43.95,
                 3.89048 )
            )
        )
    )

# HCAL front
hgchefrontDigitizer = cms.PSet( 
    accumulatorType   = cms.string("HGCDigiProducer"),
    hitCollection  = cms.string("HGCHitsHEfront"),
    digiCollection = cms.string("HGCDigisHEfront"),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    tofDelay          = cms.double(1),
    digitizationType  = cms.uint32(0),
    makeDigiSimLinks  = cms.bool(False),
    useAllChannels    = cms.bool(True),
    verbosity         = cms.untracked.uint32(0),
    digiCfg = cms.PSet(        
        keV2fC           = cms.double(0.044259), #1000 eV / 3.62 (eV per e) / 6.24150934e3 (e per fC)
        noise_fC         = cms.vdouble( [x*fC_per_ele for x in nonAgedNoises] ), #100,200,300 um
        doTimeSamples    = cms.bool(False),                                         
        feCfg   = cms.PSet( 
            # 0 only ADC, 1 ADC with pulse shape, 2 ADC+TDC with pulse shape
            fwVersion         = cms.uint32(2),
            # leakage to bunches -2, -1, in-time, +1, +2, +3 (from J. Kaplon)
            adcPulse          = cms.vdouble(0.00, 0.017,   0.817,   0.163,  0.003,  0.000), 
            pulseAvgT         = cms.vdouble(0.00, 23.42298,13.16733,6.41062,5.03946,4.5320), 
            # n bits for the ADC 
            adcNbits          = cms.uint32(10),
            # ADC saturation
            adcSaturation_fC  = cms.double(100),
            # the tdc resolution smearing (in picoseconds)
            tdcResolutionInPs = cms.double( 0.001 ),
            # LSB for TDC, assuming 12 bit dynamic range to 10 pC
            tdcNbits          = cms.uint32(12),
            # TDC saturation
            tdcSaturation_fC  = cms.double(10000),
            # raise threshold flag (~MIP/2) this is scaled 
            # for different thickness
            adcThreshold_fC   = cms.double(0.672),
            # raise usage of TDC and mode flag (from J. Kaplon)
            tdcOnset_fC       = cms.double(60) ,
            # LSB for time of arrival estimate from TDC in ns
            toaLSB_ns         = cms.double(0.005),
            #toa computation mode (0=by weighted energy, 1=simple threshold)
            toaMode           = cms.uint32(1),
            # TDC charge drain parameterisation (from J. Kaplon)
            tdcChargeDrainParameterisation = cms.vdouble(
                -919.13, 365.36, -14.10, 0.2,
                 -21.85, 49.39,  22.21,  0.8,
                 -0.28,   27.14,  43.95,
                 3.89048)
            )
        )
    )


# HCAL back (CALICE-like version, no pulse shape)
hgchebackDigitizer = cms.PSet( 
    accumulatorType   = cms.string("HGCDigiProducer"),
    hitCollection = cms.string("HcalHits"),
    digiCollection = cms.string("HGCDigisHEback"),
    maxSimHitsAccTime = cms.uint32(100),
    bxTime            = cms.double(25),
    tofDelay          = cms.double(1),
    digitizationType  = cms.uint32(1),
    makeDigiSimLinks  = cms.bool(False),
    useAllChannels    = cms.bool(True),
    verbosity         = cms.untracked.uint32(0),
    digiCfg = cms.PSet( 
        keV2MIP           = cms.double(1./1498.4),
        noise_MIP         = cms.double(0.2),
        doTimeSamples = cms.bool(False),
        nPEperMIP = cms.double(11.0),
        nTotalPE  = cms.double(1156), #1156 pixels => saturation ~600MIP
        xTalk     = cms.double(0.25),
        sdPixels  = cms.double(3.0),
        feCfg   = cms.PSet( 
            # 0 only ADC, 1 ADC with pulse shape, 2 ADC+TDC with pulse shape
            fwVersion       = cms.uint32(0),
            # n bits for the ADC 
            adcNbits        = cms.uint32(12),
            # ADC saturation : in this case we use the same variable but fC=MIP
            adcSaturation_fC = cms.double(1024),
            # threshold for digi production : in this case we use the same variable but fC=MIP
            adcThreshold_fC = cms.double(0.95)
            )
        )                              
    )

#function to set noise to aged HGCal
endOfLifeNoises = [2400.0,2250.0,1750.0]
def HGCal_setEndOfLifeNoise(digitizer):
    digitizer.digiCfg.noise_fC = cms.vdouble( [x*fC_per_ele for x in endOfLifeNoises] )
