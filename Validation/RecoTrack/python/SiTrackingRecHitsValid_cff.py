import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from Validation.RecoTrack.SiPixelTrackingRecHitsValid_cfi import *
from Validation.RecoTrack.SiStripTrackingRecHitsValid_cfi import *
from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
trackingRecHitsValid = cms.Sequence(TrackRefitter*PixelTrackingRecHitsValid*StripTrackingRecHitsValid)
TrackRefitter.TrajectoryInEvent = True

