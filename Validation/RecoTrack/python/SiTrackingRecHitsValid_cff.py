import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from Validation.RecoTrack.SiPixelTrackingRecHitsValid_cfi import *
from Validation.RecoTrack.SiStripTrackingRecHitsValid_cfi import *
trackingRecHitsValid = cms.Sequence(PixelTrackingRecHitsValid*StripTrackingRecHitsValid)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith(trackingRecHitsValid, trackingRecHitsValid.copyAndExclude([ # FIXME
    PixelTrackingRecHitsValid # Pixel validation needs to be migrated to phase1
]))
