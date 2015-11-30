import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from Validation.RecoTrack.SiPixelTrackingRecHitsValid_cfi import *
from Validation.RecoTrack.SiStripTrackingRecHitsValid_cfi import *
trackingRecHitsValid = cms.Sequence(PixelTrackingRecHitsValid*StripTrackingRecHitsValid)

# If the Phase 1 pixel detector is active, don't run this validation sequence
from Configuration.StandardSequences.Eras import eras
if eras.phase1Pixel.isChosen():
    trackingRecHitsValid.remove(PixelTrackingRecHitsValid)
