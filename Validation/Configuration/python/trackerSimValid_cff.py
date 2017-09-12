import FWCore.ParameterSet.Config as cms

# tracker validation sequences
#
from Validation.TrackerHits.trackerHitsValidation_cff import *
from Validation.TrackerDigis.trackerDigisValidation_cff import *
from Validation.TrackerRecHits.trackerRecHitsValidation_cff import *
from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.SiTrackingRecHitsValid_cff import *
from Validation.SiPixelPhase1ConfigV.SiPixelPhase1OfflineDQM_sourceV_cff import *

trackerSimValid = cms.Sequence(trackerHitsValidation+trackerDigisValidation+trackerRecHitsValidation+trackingTruthValid+trackingRecHitsValid)
trackPhase1SimValid = cms.Sequence(trackingTruthValid+siPixelPhase1OfflineDQM_sourceV)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith( trackerSimValid, trackPhase1SimValid )
