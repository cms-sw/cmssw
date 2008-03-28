import FWCore.ParameterSet.Config as cms

# magnetic field
from Geometry.CMSCommonData.cmsMagneticFieldXML_cfi import *
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
RS_MTCC_MeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cfi import *
# RoadSearchTrackCandidateMaker
rsTrackCandidatesMTCC = copy.deepcopy(rsTrackCandidates)
RS_MTCC_MeasurementTracker.ComponentName = 'RS_MTCC'
RS_MTCC_MeasurementTracker.pixelClusterProducer = ''
rsTrackCandidatesMTCC.CloudProducer = 'roadSearchCloudsMTCC'
rsTrackCandidatesMTCC.MeasurementTrackerName = 'RS_MTCC'
rsTrackCandidatesMTCC.StraightLineNoBeamSpotCloud = True
rsTrackCandidatesMTCC.HitChi2Cut = 1000.0
rsTrackCandidatesMTCC.NumHitCut = 4
rsTrackCandidatesMTCC.MinimumChunkLength = 0
rsTrackCandidatesMTCC.nFoundMin = 2

