import FWCore.ParameterSet.Config as cms

from Validation.TrackerDigis.stripDigisValidation_cfi import *
from Validation.TrackerDigis.pixelDigisValidation_cfi import *
trackerDigisValidation = cms.Sequence(pixelDigisValid*stripDigisValid)


