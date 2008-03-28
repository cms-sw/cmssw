import FWCore.ParameterSet.Config as cms

# Geometry and Magnetic Field
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from SimTracker.TrackerMaterialAnalysis.trackingMaterialAnalyser_cfi import *

