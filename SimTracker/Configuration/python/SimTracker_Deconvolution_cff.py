# Auto generated configuration file
# using: 
# $Revision: 1.45 $
# $Source: /cvs/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v $

import FWCore.ParameterSet.Config as cms
def customise(process):
    
    # Signal in Deconvolution Mode
    process.simSiStripDigis.APVpeakmode = cms.bool(False)

    # Fake Conditions Noise in Deconvolution Mode
    process.SiStripNoiseFakeESSource.NoiseStripLengthSlope = 51.
    process.SiStripNoiseFakeESSource.NoiseStripLengthQuote = 630.
    
    return(process)
