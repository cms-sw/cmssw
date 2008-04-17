import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.uniformMagneticField_cfi import *
# cms geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerP5_cff import *
import copy
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cfi import *
# RoadSearchCloudMaker
roadSearchCloudsP5 = copy.deepcopy(roadSearchClouds)
roadSearchCloudsP5.RoadsLabel = 'P5'
roadSearchCloudsP5.SeedProducer = 'roadSearchSeedsP5'
roadSearchCloudsP5.UsePixelsinRS = False
roadSearchCloudsP5.UseRphiRecHits = True
roadSearchCloudsP5.UseStereoRecHits = True
roadSearchCloudsP5.scalefactorRoadSeedWindow = 150
roadSearchCloudsP5.MinimumHalfRoad = 3.3
roadSearchCloudsP5.RPhiRoadSize = 5.
roadSearchCloudsP5.MinimalFractionOfUsedLayersPerCloud = 0.3 ##standard: 0.5

roadSearchCloudsP5.MaximalFractionOfMissedLayersPerCloud = 0.8 ##standard: 0.3

roadSearchCloudsP5.MaximalFractionOfConsecutiveMissedLayersPerCloud = 0.35 ##standard:0.15

roadSearchCloudsP5.IncreaseMaxNumberOfConsecutiveMissedLayersPerCloud = 0
roadSearchCloudsP5.IncreaseMaxNumberOfMissedLayersPerCloud = 0
roadSearchCloudsP5.StraightLineNoBeamSpotCloud = True

