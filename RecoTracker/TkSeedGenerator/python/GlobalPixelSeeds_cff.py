import FWCore.ParameterSet.Config as cms

from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelPairs_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi import *
from RecoTracker.TkSeedGenerator.GlobalPixelSeeds_cfi import *

