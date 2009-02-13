import FWCore.ParameterSet.Config as cms

# magnetic field
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
#RS_P5_MeasurementTracker = copy.deepcopy(MeasurementTracker)
#RS_P5_MeasurementTracker.ComponentName = 'RS_P5'

import copy
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cfi import *
# RoadSearchTrackCandidateMaker
rsTrackCandidatesP5 = copy.deepcopy(rsTrackCandidates)


rsTrackCandidatesP5.CloudProducer = 'roadSearchCloudsP5'
rsTrackCandidatesP5.MeasurementTrackerName = ''
rsTrackCandidatesP5.StraightLineNoBeamSpotCloud = True
rsTrackCandidatesP5.CosmicTrackMerging = True
rsTrackCandidatesP5.HitChi2Cut = 30.0
rsTrackCandidatesP5.NumHitCut = 4 ##CHANGE TO 5

rsTrackCandidatesP5.MinimumChunkLength = 2
rsTrackCandidatesP5.nFoundMin = 2

