import FWCore.ParameterSet.Config as cms

from Validation.TrackerHits.trackerHitsValidation_cfi import *

trackerHitsValidation = cms.Sequence(trackerHitsValid)
trackerSiStripHitsValidation = cms.Sequence(trackerSiStripHitsValid)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith( trackerHitsValidation, trackerSiStripHitsValidation )
