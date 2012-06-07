import FWCore.ParameterSet.Config as cms

# Silicon Strip Digitizer running with APV Mode Peak
from SimGeneral.MixingModule.stripDigitizer_cfi import *

stripDigitizer.APVpeakmode = True
