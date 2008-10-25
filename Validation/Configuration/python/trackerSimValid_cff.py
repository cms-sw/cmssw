import FWCore.ParameterSet.Config as cms

# tracker validation sequences
#
from Validation.TrackerHits.trackerHitsValidation_cff import *
from Validation.TrackerDigis.trackerDigisValidation_cff import *
from Validation.TrackerRecHits.trackerRecHitsValidation_cff import *
from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.SiTrackingRecHitsValid_cff import *
trackerSimValid = cms.Sequence(trackerHitsValidation+trackerDigisValidation+trackerRecHitsValidation+trackingTruthValid+trackingRecHitsValid)
