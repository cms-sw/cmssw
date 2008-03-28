import FWCore.ParameterSet.Config as cms

# Geometry and Magnetic Field
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from SimTracker.TrackerMaterialAnalysis.randomNumberGeneratorService_cfi import *
from SimTracker.TrackerMaterialAnalysis.trackingMaterialProducer_cfi import *

