# Auto generated configuration file
# using: 
# $Revision: 1.1 $
# $Source: /cvs_server/repositories/CMSSW/CMSSW/SimTracker/Configuration/python/SimTracker_Deconvolution_cff.py,v $

import FWCore.ParameterSet.Config as cms
def customise(process):
    # Load and Prefer the FakeConditions for SiStripNoise
    process.load("CalibTracker.SiStripESProducers.SiStripNoiseFakeSource_cfi")
    process.prefer("SiStripNoiseFakeESSource")
                                                                                          
    # Signal in Deconvolution Mode
    process.simSiStripDigis.APVpeakmode = cms.bool(False)

    # Fake Conditions Noise in Deconvolution Mode
    process.SiStripNoiseFakeESSource.NoiseStripLengthSlope = 51.
    process.SiStripNoiseFakeESSource.NoiseStripLengthQuote = 630.
    
    return(process)
