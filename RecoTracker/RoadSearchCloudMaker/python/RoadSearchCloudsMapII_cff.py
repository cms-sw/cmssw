import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# cms geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerII_cff import *
# RoadSearchCloudMaker
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cfi import *

