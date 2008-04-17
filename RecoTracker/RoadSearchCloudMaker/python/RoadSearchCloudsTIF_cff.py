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
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIF_cff import *
import copy
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cfi import *
# RoadSearchCloudMaker
roadSearchCloudsTIF = copy.deepcopy(roadSearchClouds)
roadSearchCloudsTIF.RoadsLabel = 'TIF'
roadSearchCloudsTIF.SeedProducer = 'roadSearchSeedsTIF'
roadSearchCloudsTIF.UsePixelsinRS = False
roadSearchCloudsTIF.UseRphiRecHits = True
roadSearchCloudsTIF.UseStereoRecHits = True
roadSearchCloudsTIF.scalefactorRoadSeedWindow = 150
roadSearchCloudsTIF.MinimumHalfRoad = 3.3
roadSearchCloudsTIF.RPhiRoadSize = 5.
roadSearchCloudsTIF.MinimalFractionOfUsedLayersPerCloud = 0.3 ##standard: 0.5

roadSearchCloudsTIF.MaximalFractionOfMissedLayersPerCloud = 0.8 ##standard: 0.3

roadSearchCloudsTIF.MaximalFractionOfConsecutiveMissedLayersPerCloud = 0.35 ##standard:0.15

roadSearchCloudsTIF.IncreaseMaxNumberOfConsecutiveMissedLayersPerCloud = 0
roadSearchCloudsTIF.IncreaseMaxNumberOfMissedLayersPerCloud = 0
roadSearchCloudsTIF.StraightLineNoBeamSpotCloud = True

