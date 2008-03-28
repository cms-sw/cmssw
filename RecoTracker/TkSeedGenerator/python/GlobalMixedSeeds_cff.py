import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# initialize geometry #####################
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# generate pixel seeds #####################
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4MixedPairs_cfi import *
from RecoTracker.TkSeedingLayers.MixedLayerPairs_cfi import *
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi import *

