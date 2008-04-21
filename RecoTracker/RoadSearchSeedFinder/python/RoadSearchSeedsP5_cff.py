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
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cfi import *
# RoadSearchSeedFinder
roadSearchSeedsP5 = copy.deepcopy(roadSearchSeeds)
roadSearchSeedsP5.Mode = 'STRAIGHT-LINE'
roadSearchSeedsP5.doClusterCheck = True
roadSearchSeedsP5.RoadsLabel = 'P5'
roadSearchSeedsP5.InnerSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsP5.InnerSeedRecHitAccessUseRPhi = True
roadSearchSeedsP5.InnerSeedRecHitAccessUseStereo = True
roadSearchSeedsP5.OuterSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsP5.OuterSeedRecHitAccessUseRPhi = True
roadSearchSeedsP5.OuterSeedRecHitAccessUseStereo = True

