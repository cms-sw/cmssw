import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.uniformMagneticField_cfi import *
# geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# tracker reco geometry builder
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
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
RS_P5_MeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cfi import *
# RoadSearchTrackCandidateMaker
rsTrackCandidatesP5 = copy.deepcopy(rsTrackCandidates)
RS_P5_MeasurementTracker.ComponentName = 'RS_P5'
RS_P5_MeasurementTracker.pixelClusterProducer = ''
rsTrackCandidatesP5.CloudProducer = 'roadSearchCloudsP5'
rsTrackCandidatesP5.MeasurementTrackerName = 'RS_P5'
rsTrackCandidatesP5.StraightLineNoBeamSpotCloud = True
rsTrackCandidatesP5.CosmicTrackMerging = True
rsTrackCandidatesP5.HitChi2Cut = 30.0
rsTrackCandidatesP5.NumHitCut = 4 ##CHANGE TO 5

rsTrackCandidatesP5.MinimumChunkLength = 2
rsTrackCandidatesP5.nFoundMin = 2

