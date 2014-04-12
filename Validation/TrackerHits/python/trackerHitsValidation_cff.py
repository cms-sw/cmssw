import FWCore.ParameterSet.Config as cms

from Validation.TrackerHits.trackerHitsValidation_cfi import *
trackerHitsValidation = cms.Sequence(trackerHitsValid)

