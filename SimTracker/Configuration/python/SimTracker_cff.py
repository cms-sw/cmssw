import FWCore.ParameterSet.Config as cms

from SimTracker.SiStripDigitizer.SiStripDigiSimLink_cfi import simSiStripDigiSimLink

# Combined sequence
trDigi = cms.Sequence(simSiStripDigiSimLink)

import SimGeneral.MixingModule.stripDigitizer_APVModeDec_cff 
