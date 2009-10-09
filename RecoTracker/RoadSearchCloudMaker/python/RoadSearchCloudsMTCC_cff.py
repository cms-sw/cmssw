import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.uniformMagneticField_cfi import *
# cms geometry
from Geometry.CMSCommonData.cmsMTCCGeometryXML_cfi import *
# tracker geometry
# tracker numbering
# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerMTCC_cff import *
import RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cfi
# RoadSearchCloudMaker
roadSearchCloudsMTCC = RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cfi.roadSearchClouds.clone()
roadSearchCloudsMTCC.RoadsLabel = 'MTCC'
roadSearchCloudsMTCC.SeedProducer = 'roadSearchSeedsMTCC'
roadSearchCloudsMTCC.UsePixelsinRS = False
roadSearchCloudsMTCC.UseRphiRecHits = True
roadSearchCloudsMTCC.UseStereoRecHits = True
roadSearchCloudsMTCC.scalefactorRoadSeedWindow = 15
roadSearchCloudsMTCC.MinimumHalfRoad = 0.50
roadSearchCloudsMTCC.RPhiRoadSize = 5.
roadSearchCloudsMTCC.MinimalNumberOfUsedLayersPerRoad = 3
roadSearchCloudsMTCC.MaximalNumberOfMissedLayersPerRoad = 3
roadSearchCloudsMTCC.StraightLineNoBeamSpotCloud = True

