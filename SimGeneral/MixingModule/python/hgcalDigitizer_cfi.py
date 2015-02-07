import FWCore.ParameterSet.Config as cms

# Base configurations for HGCal digitizers

# ECAL
hgceeDigitizer = cms.PSet( accumulatorType   = cms.string("HGCDigiProducer"),
                           hitCollection     = cms.string("HGCHitsEE"),
                           digiCollection    = cms.string("HGCDigisEE"),
                           maxSimHitsAccTime = cms.uint32(100),
                           bxTime            = cms.double(25),
                           tofDelay          = cms.double(1),
                           digitizationType  = cms.uint32(0),
                           makeDigiSimLinks  = cms.bool(False),
                           useAllChannels    = cms.bool(True),
                           verbosity         = cms.untracked.uint32(0),
                           digiCfg = cms.PSet( mipInKeV         = cms.double(55.1),                                               
                                               mipInfC          = cms.double(2.35),
                                               mip2noise        = cms.double(7.0),
                                               doTimeSamples    = cms.bool(False),                                         
                                               feCfg   = cms.PSet( # 0 only ADC, 1 ADC with pulse shape, 2 ADC+TDC with pulse shape
                                                                   fwVersion         = cms.uint32(2),
                                                                   # leakage to bunches -2, -1, in-time, +1, +2, +3 (from J. Kaplon)
                                                                   adcPulse          = cms.vdouble(0.00,0.017,0.817,0.163,0.003,0.000), 
                                                                   # n bits for the ADC 
                                                                   adcNbits          = cms.uint32(10),
                                                                   # ADC saturation
                                                                   adcSaturation_fC  = cms.double(100),
                                                                   # LSB for TDC, assuming 12 bit dynamic range to 10 pC
                                                                   tdcNbits          = cms.uint32(12),
                                                                   # TDC saturation
                                                                   tdcSaturation_fC  = cms.double(10000),
                                                                   # raise threshold flag (~MIP/2)
                                                                   adcThreshold_fC   = cms.double(1.175),
                                                                   # raise usage of TDC and mode flag (from J. Kaplon)
                                                                   tdcOnset_fC       = cms.double(60) ,
                                                                   # LSB for time of arrival estimate from TDC in ns
                                                                   toaLSB_ns         = cms.double(0.005),
                                                                   # TDC charge drain parameterisation (from J. Kaplon)
                                                                   tdcChargeDrainParameterisation = cms.vdouble(200.0,
                                                                                                               139.979, 0.779, 0.000,
                                                                                                               32.624,  0.894, 32.215,
                                                                                                               8.53)
                                                                   )
                                               )
                           )

# HCAL front
hgchefrontDigitizer = cms.PSet( accumulatorType   = cms.string("HGCDigiProducer"),
                                hitCollection  = cms.string("HGCHitsHEfront"),
                                digiCollection = cms.string("HGCDigisHEfront"),
                                maxSimHitsAccTime = cms.uint32(100),
                                bxTime            = cms.double(25),
                                tofDelay          = cms.double(1),
                                digitizationType  = cms.uint32(0),
                                makeDigiSimLinks  = cms.bool(False),
                                useAllChannels    = cms.bool(True),
                                verbosity         = cms.untracked.uint32(0),
                                digiCfg = cms.PSet( mipInKeV         = cms.double(85.0),                                               
                                                    mipInfC          = cms.double(3.52),
                                                    mip2noise        = cms.double(7.0),
                                                    doTimeSamples    = cms.bool(False),                                         
                                                    feCfg   = cms.PSet( # 0 only ADC, 1 ADC with pulse shape, 2 ADC+TDC with pulse shape
                                                                        fwVersion         = cms.uint32(2),
                                                                        # leakage to bunches -2, -1, in-time, +1, +2, +3 (from J. Kaplon)
                                                                        adcPulse          = cms.vdouble(0.00,0.017,0.817,0.163,0.003,0.000), 
                                                                        # n bits for the ADC 
                                                                        adcNbits          = cms.uint32(10),
                                                                        # ADC saturation
                                                                        adcSaturation_fC  = cms.double(100),
                                                                        # LSB for TDC, assuming 12 bit dynamic range to 10 pC
                                                                        tdcNbits          = cms.uint32(12),
                                                                        # TDC saturation
                                                                        tdcSaturation_fC  = cms.double(10000),
                                                                        # raise threshold flag (~MIP/2)
                                                                        adcThreshold_fC   = cms.double(1.76),
                                                                        # raise usage of TDC and mode flag (from J. Kaplon)
                                                                        tdcOnset_fC       = cms.double(60) ,
                                                                        # LSB for time of arrival estimate from TDC in ns
                                                                        toaLSB_ns         = cms.double(0.005),
                                                                        # TDC charge drain parameterisation (from J. Kaplon)
                                                                        tdcChargeDrainParameterisation = cms.vdouble(200.0,
                                                                                                                     139.979, 0.779, 0.000,
                                                                                                                     32.624,  0.894, 32.215,
                                                                                                                     8.53)
                                                                        )
                                                    )
                                )


# HCAL back (CALICE-like version, no pulse shape)
hgchebackDigitizer = cms.PSet( accumulatorType   = cms.string("HGCDigiProducer"),
                               hitCollection = cms.string("HGCHitsHEback"),
                               digiCollection = cms.string("HGCDigisHEback"),
                               maxSimHitsAccTime = cms.uint32(100),
                               bxTime            = cms.double(25),
                               tofDelay          = cms.double(1),
                               digitizationType  = cms.uint32(1),
                               makeDigiSimLinks  = cms.bool(False),
                               useAllChannels    = cms.bool(True),
                               verbosity         = cms.untracked.uint32(0),
                               digiCfg = cms.PSet( mipInKeV = cms.double(1498.4),
                                                   mip2noise = cms.double(5.0),                                                   
                                                   doTimeSamples = cms.bool(False),
                                                   feCfg   = cms.PSet( fwVersion      = cms.uint32(0),
                                                                       adcThreshold  = cms.double(4),
                                                                       lsbInMIP = cms.double(0.25),
                                                                       shaperN       = cms.double(1.),
                                                                       shaperTau     = cms.double(0.) ),
                                                   caliceSpecific = cms.PSet( nPEperMIP = cms.double(11.0),
                                                                              #1156 pixels => saturation ~600MIP
                                                                              nTotalPE  = cms.double(11560),
                                                                              xTalk     = cms.double(0.25),
                                                                              sdPixels  = cms.double(3.0) )
                                                   )
                               )




                           


