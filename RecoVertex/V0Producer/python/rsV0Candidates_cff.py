import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# tracker geometry
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# global tracker geometry
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# rsV0Producer
from RecoVertex.V0Producer.rsV0Candidates_cfi import *

