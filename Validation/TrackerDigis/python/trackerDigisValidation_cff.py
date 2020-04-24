import FWCore.ParameterSet.Config as cms

from Validation.TrackerDigis.stripDigisValidation_cfi import *
from Validation.TrackerDigis.pixelDigisValidation_cfi import *

trackerDigisValidation = cms.Sequence(pixelDigisValid*stripDigisValid)
trackerDigisStripValidation = cms.Sequence(stripDigisValid)
  		  
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith( trackerDigisValidation, trackerDigisStripValidation )


