import FWCore.ParameterSet.Config as cms

# Silicon Strip Digitizer running with APV Mode Deconvolution
from SimTracker.SiStripDigitizer.SiStripDigi_cfi import *
simSiStripDigis.APVpeakmode = False
simSiStripDigis.electronPerAdc = 262.0 #this is the value measured in peak... should we add 15%?

