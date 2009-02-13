import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.uniformMagneticField_cfi import *
# geometry
# tracker geometry
# tracker numbering
# tracker reco geometry builder

# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
# MeasurementTracker
RS_TIF_MeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cfi import *
# RoadSearchTrackCandidateMaker
rsTrackCandidatesTIF = copy.deepcopy(rsTrackCandidates)
RS_TIF_MeasurementTracker.ComponentName = 'RS_TIF'
RS_TIF_MeasurementTracker.pixelClusterProducer = ''
rsTrackCandidatesTIF.CloudProducer = 'roadSearchCloudsTIF'
rsTrackCandidatesTIF.MeasurementTrackerName = 'RS_TIF'
rsTrackCandidatesTIF.StraightLineNoBeamSpotCloud = True
rsTrackCandidatesTIF.HitChi2Cut = 30.0
rsTrackCandidatesTIF.NumHitCut = 4 ##CHANGE TO 5

rsTrackCandidatesTIF.MinimumChunkLength = 2
rsTrackCandidatesTIF.nFoundMin = 2

