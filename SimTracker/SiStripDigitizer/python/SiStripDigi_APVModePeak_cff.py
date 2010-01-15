import FWCore.ParameterSet.Config as cms

# Silicon Strip Digitizer running with APV Mode Peak
from SimTracker.SiStripDigitizer.SiStripDigi_cfi import *
simSiStripDigis.APVpeakmode = True
simSiStripDigis.electronPerAdc = 262.0 #from CRAFT08; see CFT-09-002 and https://twiki.cern.ch/twiki/bin/view/CMS/TRKTuningPeakMC
