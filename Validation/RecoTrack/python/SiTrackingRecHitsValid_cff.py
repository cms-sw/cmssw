import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from Validation.RecoTrack.SiPixelTrackingRecHitsValid_cfi import *
from Validation.RecoTrack.SiStripTrackingRecHitsValid_cfi import *
trackingRecHitsValid = cms.Sequence(PixelTrackingRecHitsValid*StripTrackingRecHitsValid)
eras.phase1Pixel.toReplaceWith(trackingRecHitsValid, trackingRecHitsValid.copyAndExclude([ # FIXME
    PixelTrackingRecHitsValid # Pixel validation needs to be migrated to phase1
]))
