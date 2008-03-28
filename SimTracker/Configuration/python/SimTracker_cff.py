import FWCore.ParameterSet.Config as cms

# Tracker Digitization 
# (modelng of the electronics response in pixels and sistrips)
#
# Pixel's digitization
#
from SimTracker.SiPixelDigitizer.PixelDigi_cfi import *
# SiStrip's digitization default in APV Mode Peak
#
#include "SimTracker/SiStripDigitizer/data/SiStripDigi_APVModeDec.cff"
from SimTracker.SiStripDigitizer.SiStripDigi_APVModePeak_cff import *
trDigi = cms.Sequence(siPixelDigis+siStripDigis)

