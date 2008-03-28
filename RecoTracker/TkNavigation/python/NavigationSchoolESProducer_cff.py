import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# initialize geometry #####################
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# navigationschools
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi import *
from RecoTracker.TkNavigation.BeamHaloNavigationSchoolESProducer_cfi import *
from RecoTracker.TkNavigation.CosmicsNavigationSchoolESProducer_cfi import *

