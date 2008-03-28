import FWCore.ParameterSet.Config as cms

# magnetic field
from Geometry.CMSCommonData.cmsMagneticFieldXML_cfi import *
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
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cfi import *
# RoadSearchSeedFinder
roadSearchSeedsTIF = copy.deepcopy(roadSearchSeeds)
roadSearchSeedsTIF.Mode = 'STRAIGHT-LINE'
roadSearchSeedsTIF.CosmicTracking = True
roadSearchSeedsTIF.RoadsLabel = 'TIF'
roadSearchSeedsTIF.InnerSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsTIF.InnerSeedRecHitAccessUseRPhi = True
roadSearchSeedsTIF.InnerSeedRecHitAccessUseStereo = True
roadSearchSeedsTIF.OuterSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsTIF.OuterSeedRecHitAccessUseRPhi = True
roadSearchSeedsTIF.OuterSeedRecHitAccessUseStereo = True

