import FWCore.ParameterSet.Config as cms

# Silicon Strip Digitizer running with APV Mode Deconvolution
from SimTracker.SiStripDigitizer.SiStripDigi_cfi import *
simSiStripDigis.APVpeakmode = False
simSiStripDigis.electronPerAdc = 217.0 # see https://twiki.cern.ch/twiki/bin/view/CMS/TRKTuningDecoMC

