import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from Validation.RecoTrack.SiPixelTrackingRecHitsValid_cfi import *
from Validation.RecoTrack.SiStripTrackingRecHitsValid_cfi import *
trackingRecHitsValid = cms.Sequence(PixelTrackingRecHitsValid*StripTrackingRecHitsValid)

