import FWCore.ParameterSet.Config as cms

# Tracker Digitization 
# (modeling of the electronics response in pixels and sistrips)

# Pixel's digitization
from SimTracker.SiPixelDigitizer.PixelDigi_cfi import *
# SiStrip's digitization in APV Mode Peak
from SimTracker.SiStripDigitizer.SiStripDigi_APVModePeak_cff import *

# Combined sequence
trDigi = cms.Sequence(simSiPixelDigis+simSiStripDigis)
