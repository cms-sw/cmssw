import FWCore.ParameterSet.Config as cms

# Base configurations for HGCal digitizers

# ECAL
hgceeDigitizer = cms.PSet( accumulatorType   = cms.string("HGCDigiProducer"),
                           hitCollection     = cms.string("HGCHitsEE"),
                           digiCollection    = cms.string("HGCDigisEE"),
                           maxSimHitsAccTime = cms.uint32(100),
                           bxTime            = cms.int32(25),
                           tofDelay          = cms.double(1),
                           digitizationType  = cms.uint32(0),
                           makeDigiSimLinks  = cms.bool(False),
                           useAllChannels    = cms.bool(True),
                           verbosity         = cms.untracked.int32(0),
                           digiCfg = cms.PSet( mipInKeV      = cms.double(55.1),
                                               lsbInMIP      = cms.double(0.25),
                                               mip2noise     = cms.double(7.0),
                                               adcThreshold  = cms.uint32(2),
                                               doTimeSamples = cms.bool(False),
                                               shaperN       = cms.double(1.),
                                               shaperTau     = cms.double(0.)
                                               )
                           )

# HCAL front
hgchefrontDigitizer = cms.PSet( accumulatorType   = cms.string("HGCDigiProducer"),
                                hitCollection  = cms.string("HGCHitsHEfront"),
                                digiCollection = cms.string("HGCDigisHEfront"),
                                maxSimHitsAccTime = cms.uint32(100),
                                bxTime            = cms.int32(25),
                                tofDelay          = cms.double(1),
                                digitizationType  = cms.uint32(0),
                                makeDigiSimLinks  = cms.bool(False),
                                useAllChannels    = cms.bool(True),
                                verbosity         = cms.untracked.int32(0),
                                digiCfg = cms.PSet( mipInKeV      = cms.double(85.0),
                                                    lsbInMIP      = cms.double(0.25),
                                                    mip2noise     = cms.double(7.0),
                                                    adcThreshold  = cms.uint32(2),
                                                    doTimeSamples = cms.bool(False),
                                                    shaperN       = cms.double(1.),
                                                    shaperTau     = cms.double(0.)
                                                    )
                                )
                                                    

# HCAL back
hgchebackDigitizer = cms.PSet( accumulatorType   = cms.string("HGCDigiProducer"),
                               hitCollection = cms.string("HGCHitsHEback"),
                               digiCollection = cms.string("HGCDigisHEback"),
                               maxSimHitsAccTime = cms.uint32(100),
                               bxTime            = cms.int32(25),
                               tofDelay          = cms.double(1),
                               digitizationType  = cms.uint32(1),
                               makeDigiSimLinks  = cms.bool(False),
                               useAllChannels    = cms.bool(True),
                               verbosity         = cms.untracked.int32(0),
                               digiCfg = cms.PSet( mipInKeV = cms.double(1498.4),
                                                   lsbInMIP = cms.double(0.25),
                                                   mip2noise = cms.double(5.0),
                                                   adcThreshold  = cms.uint32(4),
                                                   doTimeSamples = cms.bool(False),
                                                   shaperN       = cms.double(1.),
                                                   shaperTau     = cms.double(0.),
                                                   caliceSpecific = cms.PSet( nPEperMIP = cms.double(11.0),
                                                                              #1156 pixels => saturation ~600MIP
                                                                              nTotalPE  = cms.double(11560),
                                                                              xTalk     = cms.double(0.25),
                                                                              sdPixels  = cms.double(3.0) )
                                                   )
                               )



                           


